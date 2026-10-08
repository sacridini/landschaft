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

    var R = 6378137, MAXPX = 4096;

    function Live(map, id, layers) {
        this.map = map; this.id = id; this.layers = layers; this.live = false;
        var self = this, wait = null;
        maps.push(this);
        this.schedule = function () {
            if (!self.live) { return; }
            clearTimeout(wait);
            wait = setTimeout(function () { self.layers.forEach(function (l) { self.update(l); }); }, 200);
        };
        map.on("moveend zoomend overlayadd overlayremove resize", this.schedule);
        map.on("layeradd layerremove", function (e) {  // a band of a stack shown (not a detail drawn)
            if (!(e.layer && e.layer._lsIsDetail)) { self.schedule(); }
        });
        if (via) { setTimeout(function () { self.hello(); }, 0); }
    }
    Live.prototype.hello = function () {
        var self = this;
        ask({op: "hello", map: this.id}).then(function (a) {
            if (a && a.ok) { self.live = true; self.schedule(); }
        });
    };
    // the view in Web Mercator and the size of a screen pixel there
    Live.prototype.view = function () {
        var map = this.map, b = map.getPixelBounds(), z = map.getZoom(), m = window.devicePixelRatio || 1;
        var nw = L.CRS.EPSG3857.project(map.unproject(b.min, z)), se = L.CRS.EPSG3857.project(map.unproject(b.max, z));
        var v = [nw.x, se.y, se.x, nw.y];
        return {box: v, px: (v[2] - v[0]) / (map.getSize().x * m)};
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

    Live.prototype.update = function (l) {
        var base = window[l.v], map = this.map, self = this;
        if (!base || !map.hasLayer(base)) { this.clear(l); return; }
        var v = this.view(), box = inter(v.box, l.ext);
        if (!box || l.res <= v.px * 1.5) { this.clear(l); return; }  // the overview is fine enough here
        var d = l.det;
        if (d && inside(box, d.box) && d.res <= v.px * 1.5 && d.res >= v.px / 3) { return; }  // still good
        if (d && !inside(box, d.box)) {  // panned beyond the detail: the overview again until the next
            if (l.kind === "gl") { this.release(l); } else { hide(base, false); }
        }
        // the view and half of it around (pans), at the screen's resolution
        var dx = (box[2] - box[0]) * 0.5, dy = (box[3] - box[1]) * 0.5;
        var want = inter([box[0] - dx, box[1] - dy, box[2] + dx, box[3] + dy], l.ext);
        var px = Math.max(v.px, (want[2] - want[0]) / MAXPX, (want[3] - want[1]) / MAXPX);
        var w = Math.max(1, Math.round((want[2] - want[0]) / px)), h = Math.max(1, Math.round((want[3] - want[1]) / px));
        var req = l.req = (l.req || 0) + 1;
        ask({op: "detail", map: this.id, key: l.key, bbox: want, w: w, h: h}).then(function (a) {
            if (req !== l.req || !map.hasLayer(base)) { return; }  // a later view asked meanwhile
            if (!a || !a.ok) { return; }
            if (l.kind === "gl") { self.showGL(l, base, a, want, px); } else { self.showImage(l, base, a, want, px); }
        });
    };

    Live.prototype.showImage = function (l, base, a, box, px) {
        var map = this.map, old = l.det;
        var o = L.imageOverlay(a.url, bounds(a.ext), {opacity: base.options.opacity, interactive: false,
                                                       pane: base.options.pane || "overlayPane", zIndex: 3});
        o._lsIsDetail = true;
        o.addTo(map);
        l.det = {layer: o, box: box, res: px};
        base._lsDetail = {layer: o, ext: a.ext, palette: a.palette || null, lookup: a.lookup || null};
        if (old) { map.removeLayer(old.layer); }
        o.once("load", function () { if (l.det && l.det.layer === o) { hide(base, true); } });
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
        var map = this.map;
        this.release(l);
        var g = new L.LsGL(a.spec, a.data, base.options.pane ? {pane: base.options.pane} : {});
        g._lsBase = base;
        g._lsIsDetail = true;
        g._fill = base._fill; g._line = base._line;
        g.addTo(map);
        l.det = {layer: g, box: box, res: px};
        g._ready.then(function () {
            if (l.det && l.det.layer === g) { base._suppress = true; base._draw(); }
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

    Live.prototype.release = function (l) {
        var d = l.det, base = window[l.v];
        if (!d) { return; }
        l.det = null;
        this.map.removeLayer(d.layer);
        if (l.kind === "gl") {
            var i = (window.LS_GL || []).indexOf(d.layer);
            if (i >= 0) { window.LS_GL.splice(i, 1); }
            if (base) { base._suppress = false; base._draw(); }
        } else if (base) {
            base._lsDetail = null;
            hide(base, false);
        }
    };
    Live.prototype.clear = function (l) { l.req = (l.req || 0) + 1; this.release(l); };

    window.LS_LIVE = {add: function (map, id, layers) { return new Live(map, id, layers); }};
})();
