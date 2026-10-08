/* landschaft (GPL-3.0-or-later): detail on demand while Python runs (see _live.py).
 *
 * As a map is zoomed in past the resolution of the overview a layer holds, the part in view is
 * asked of Python at the screen's resolution: through pywebview's JavaScript API in a window, or
 * through a widget of the notebook (messages to the page that holds this frame). Without either
 * (a page saved and opened elsewhere) nothing is asked and the overview stays.
 */
(function () {
    "use strict";
    if (window.LS_LIVE) { return; }
    var token = Math.random().toString(36).slice(2), seq = 0, waiting = {}, via = null, maps = [];

    function ask(msg) {
        if (via === "webview") { return window.pywebview.api.ls_live(msg); }
        if (via === "parent") {
            return new Promise(function (resolve) {
                var req = token + ":" + (++seq);
                waiting[req] = resolve;
                msg.req = req;
                window.parent.postMessage({lsLive: msg}, "*");
                setTimeout(function () {  // no answer (the kernel busy or gone): give up on this one
                    if (waiting[req]) { delete waiting[req]; resolve({ok: false}); }
                }, 120000);
            });
        }
        return Promise.resolve({ok: false});
    }
    window.addEventListener("message", function (e) {
        var d = e.data;
        if (d && d.lsLiveReply && waiting[d.lsLiveReply.req]) {
            var r = waiting[d.lsLiveReply.req];
            delete waiting[d.lsLiveReply.req];
            r(d.lsLiveReply.data || {ok: false});
        } else if (d && d.lsLiveHere) {
            hello();
        }
    });

    var greeted = false;
    function start(how) {
        if (via) { return; }
        via = how;
        maps.forEach(function (m) { m.hello(); });
    }
    function hello() {  // a notebook: ask the page above whether a kernel answers for these maps
        if (via || window.parent === window || greeted) { return; }
        maps.forEach(function (m) {
            var req = token + ":" + (++seq);
            waiting[req] = function (a) { if (a && a.ok) { greeted = true; start("parent"); } };
            window.parent.postMessage({lsLive: {op: "hello", map: m.id, req: req}}, "*");
        });
    }
    if (window.pywebview && window.pywebview.api && window.pywebview.api.ls_live) { via = "webview"; }
    window.addEventListener("pywebviewready", function () {
        if (window.pywebview.api && window.pywebview.api.ls_live) { start("webview"); }
    });
    var tries = 0;
    var timer = setInterval(function () {
        if (via || ++tries > 20) { clearInterval(timer); return; }
        hello();
    }, 750);

    var MAXPX = 4096, KEEP = 12;  // the details of the last bands seen, kept for their view

    function Live(map, id, layers) {
        this.map = map; this.id = id; this.layers = layers; this.live = false; this.kept = [];
        var self = this, wait = null;
        maps.push(this);
        this.schedule = function () {
            if (!self.live) { return; }
            clearTimeout(wait);
            wait = setTimeout(function () { self.layers.forEach(function (l) { self.update(l); }); }, 100);
        };
        map.on("moveend zoomend overlayadd overlayremove resize", this.schedule);
        // a zoom asks for the detail of the view it goes to as it starts, not once it has ended
        map.on("zoomanim", function (e) {
            if (!self.live) { return; }
            var v = self.view(e.center, e.zoom);
            self.layers.forEach(function (l) { self.update(l, v); });
        });
        map.on("layerremove", function (e) {  // a band of a stack taken off: its detail with it, at once
            var l = self.entry(e.layer);
            if (l) { self.clear(l); }
        });
        map.on("layeradd", function (e) {  // a band put on: its detail, made ready beforehand, at once
            var l = self.entry(e.layer);
            if (!l) { return; }
            var r = l.ready, v = self.view(), k = l.kept;
            l.ready = null;
            if (!(r && self.fits(r, v, l)) && k && self.fits(k, v, l)) { r = k; }
            if (r && self.fits(r, v, l)) { self.showImage(l, e.layer, r.a, r.box, r.res, r.img); }
            self.schedule();
        });
        if (via) { setTimeout(function () { self.hello(); }, 0); }
    }
    Live.prototype.entry = function (layer) {
        for (var i = 0; i < this.layers.length; i++) { if (window[this.layers[i].v] === layer) { return this.layers[i]; } }
        return null;
    };
    Live.prototype.hello = function () {
        var self = this;
        ask({op: "hello", map: this.id}).then(function (a) {
            if (a && a.ok) { self.live = true; self.schedule(); }
        });
    };
    // the view in Web Mercator and the size of a screen pixel there: the map's, or that of a zoom
    // to (center, zoom)
    Live.prototype.view = function (center, zoom) {
        var map = this.map, m = window.devicePixelRatio || 1, size = map.getSize(), z = zoom === undefined ? map.getZoom() : zoom;
        var c = map.project(center || map.getCenter(), z), half = size.divideBy(2);
        var nw = L.CRS.EPSG3857.project(map.unproject(c.subtract(half), z));
        var se = L.CRS.EPSG3857.project(map.unproject(c.add(half), z));
        var v = [nw.x, se.y, se.x, nw.y];
        return {box: v, px: (v[2] - v[0]) / (size.x * m)};
    };
    function inter(a, b) {
        var o = [Math.max(a[0], b[0]), Math.max(a[1], b[1]), Math.min(a[2], b[2]), Math.min(a[3], b[3])];
        return (o[2] > o[0] && o[3] > o[1]) ? o : null;
    }
    function inside(a, b) { return a[0] >= b[0] && a[1] >= b[1] && a[2] <= b[2] && a[3] <= b[3]; }
    // the overview hidden under its detail (both half transparent, it would show through)
    function hide(base, hidden) {
        base._lsHidden = hidden;
        var el = base.getElement && base.getElement();
        if (el) { el.style.opacity = hidden ? 0 : base.options.opacity; }
    }
    function bounds(e) {
        var sw = L.CRS.EPSG3857.unproject(L.point(e[0], e[1])), ne = L.CRS.EPSG3857.unproject(L.point(e[2], e[3]));
        return L.latLngBounds(sw, ne);
    }
    // a detail (or a question) serves a view: it covers what of the layer is in view, finely enough
    Live.prototype.fits = function (d, v, l) {
        var box = inter(v.box, (l || {}).ext || d.box) || v.box;
        return inside(box, d.box) && d.res <= v.px * 1.5 && d.res >= v.px / 3;
    };
    // what to ask for a view: the part in view and a quarter of it around (pans), at the screen's
    // resolution (more around takes longer to draw: the detail would come after the zoom ends)
    Live.prototype.plan = function (l, v) {
        var box = inter(v.box, l.ext);
        if (!box || l.res <= v.px * 1.5) { return null; }  // the overview is fine enough here
        var dx = (box[2] - box[0]) * 0.25, dy = (box[3] - box[1]) * 0.25;
        var want = inter([box[0] - dx, box[1] - dy, box[2] + dx, box[3] + dy], l.ext);
        var px = Math.max(v.px, (want[2] - want[0]) / MAXPX, (want[3] - want[1]) / MAXPX);
        return {box: want, res: px, w: Math.max(1, Math.round((want[2] - want[0]) / px)),
                h: Math.max(1, Math.round((want[3] - want[1]) / px))};
    };
    function decoded(a) {  // the image of an answer, decoded (drawn at once when shown)
        var img = new Image();
        img.src = a.url;
        return (img.decode ? img.decode() : Promise.resolve()).then(function () { return img; }, function () { return img; });
    }

    Live.prototype.update = function (l, target) {
        var base = window[l.v], map = this.map, self = this;
        if (!base || !map.hasLayer(base)) { this.clear(l); return; }
        var v = target || this.view(), p = this.plan(l, v);
        if (!p) {
            if (!target) { this.clear(l); }
            return;
        }
        if (l.det && this.fits(l.det, v, l)) { return; }  // still good
        if (l.asked && this.fits(l.asked, v, l)) { return; }  // on its way
        if (!target && l.det && !inside(inter(v.box, l.ext), l.det.box) && l.kind !== "gl") {
            hide(base, false);  // panned beyond the detail: the overview again until the next
        }
        var req = l.req = (l.req || 0) + 1;
        l.asked = p;
        ask({op: "detail", map: this.id, key: l.key, bbox: p.box, w: p.w, h: p.h}).then(function (a) {
            if (req !== l.req) { return; }  // a later view asked meanwhile
            l.asked = null;
            if (!a || !a.ok || !map.hasLayer(base)) { return; }
            var show = function (img) {
                if (req !== l.req || !map.hasLayer(base)) { return; }
                if (map._animatingZoom) { map.once("zoomend", function () { show(img); }); return; }
                if (l.kind === "gl") { self.showGL(l, base, a, p.box, p.res); }
                else { self.showImage(l, base, a, p.box, p.res, img); }
            };
            if (l.kind === "gl") { show(null); } else { decoded(a).then(show); }
        });
    };

    // the detail of a layer not on the map yet (the next band of a stack), for the view: the band is
    // put on with it, never coarse first. Resolves when it is ready, at most after ``wait`` ms.
    Live.prototype.prepare = function (l, wait) {
        if (!this.live || l.kind !== "image") { return Promise.resolve(); }
        var v = this.view(), p = this.plan(l, v);
        if (!p) { return Promise.resolve(); }
        if (l.det && this.fits(l.det, v, l)) { return Promise.resolve(); }
        if (!(l.pre && this.fits(l.pre.plan, v, l))) {
            var pre = {plan: p};
            pre.promise = ask({op: "detail", map: this.id, key: l.key, bbox: p.box, w: p.w, h: p.h}).then(function (a) {
                if (!a || !a.ok) { return; }
                return decoded(a).then(function (img) {
                    if (l.pre === pre) { l.ready = {a: a, img: img, box: p.box, res: p.res}; }
                });
            });
            l.pre = pre;
        }
        var done = l.pre.promise;
        return wait ? Promise.race([done, new Promise(function (r) { setTimeout(r, wait); })]) : done;
    };

    Live.prototype.showImage = function (l, base, a, box, px, img) {
        var map = this.map, old = l.det;
        var o = L.imageOverlay(img || a.url, bounds(a.ext), {opacity: base.options.opacity, interactive: false, className: "ls-detail",
                                                             pane: base.options.pane || "overlayPane", zIndex: 3});
        o._lsIsDetail = true;
        l.det = {layer: o, box: box, res: px, a: a};
        base._lsDetail = {layer: o, ext: a.ext, palette: a.palette || null, lookup: a.lookup || null};
        var swap = function () {  // the old detail off once the new one is drawn: never a gap
            if (old) { map.removeLayer(old.layer); }
            if (l.det && l.det.layer === o) { hide(base, true); }
        };
        o.addTo(map);
        var el = o.getElement();
        if (el && el.complete && el.naturalWidth) { swap(); } else { o.once("load", swap); }
        if (!base._lsWrapped) {  // the detail follows the layer's opacity
            base._lsWrapped = true;
            var set = base.setOpacity;
            base.setOpacity = function (op) {
                set.call(base, op);
                if (l.det) { l.det.layer.setOpacity(op); }
                if (base._lsHidden) { hide(base, true); }
                return base;
            };
        }
    };

    Live.prototype.showGL = function (l, base, a, box, px) {
        var map = this.map, old = l.det;
        var g = new L.LsGL(a.spec, a.data, base.options.pane ? {pane: base.options.pane} : {});
        g._lsBase = base;
        g._lsIsDetail = true;
        g._fill = base._fill; g._line = base._line;
        g.addTo(map);
        l.det = {layer: g, box: box, res: px};
        g._ready.then(function () {  // the old detail off once the new one is drawn
            if (!(l.det && l.det.layer === g)) { return; }
            if (old) { dropGL(map, old.layer); }
            base._suppress = true; base._draw();
        });
        if (!base._lsWrapped) {
            base._lsWrapped = true;
            var set = base.setOpacity;
            base.setOpacity = function (op) {
                set.call(base, op);
                if (l.det) { l.det.layer.setOpacity(op); }
                return base;
            };
        }
    };
    function dropGL(map, layer) {
        map.removeLayer(layer);
        var i = (window.LS_GL || []).indexOf(layer);
        if (i >= 0) { window.LS_GL.splice(i, 1); }
    }

    Live.prototype.release = function (l) {
        var d = l.det, base = window[l.v];
        if (!d) { return; }
        l.det = null;
        if (l.kind === "gl") {
            dropGL(this.map, d.layer);
            if (base) { base._suppress = false; base._draw(); }
        } else {
            this.map.removeLayer(d.layer);
            if (base) { base._lsDetail = null; hide(base, false); }
            this.keep(l, d);
        }
    };
    // a detail taken off (its band left) kept for a while: the band shows sharp at once if it comes back
    Live.prototype.keep = function (l, d) {
        var img = d.layer.getElement && d.layer.getElement();
        if (!img || !img.complete || !d.a) { return; }
        l.kept = {a: d.a, img: img, box: d.box, res: d.res};
        var i = this.kept.indexOf(l);
        if (i >= 0) { this.kept.splice(i, 1); }
        this.kept.push(l);
        while (this.kept.length > KEEP) { this.kept.shift().kept = null; }
    };
    Live.prototype.clear = function (l) { l.req = (l.req || 0) + 1; l.asked = null; this.release(l); };

    window.LS_LIVE = {
        add: function (map, id, layers) { return new Live(map, id, layers); },
        // the detail of a layer about to be shown, made ready first (see Live.prepare)
        prepare: function (layer, wait) {
            for (var i = 0; i < maps.length; i++) {
                var l = maps[i].entry(layer);
                if (l) { return maps[i].prepare(l, wait); }
            }
            return Promise.resolve();
        }
    };
})();
