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

    var MAXPX = 4096;
    // details kept: every one seen or asked ahead (its PNG, ~0.1 to 0.3 MB), and decoded (drawn at
    // once, 4 bytes a pixel) up to this many bytes, the most recent and the bands around the one shown
    var DECODED = 256e6, AHEAD = 8;  // bands asked for at once ahead (the nearest first)

    function Live(map, id, layers) {
        this.map = map; this.id = id; this.layers = layers; this.live = false;
        this.decoded = []; this.bytes = 0; this.gen = 0;
        var self = this, wait = null;
        maps.push(this);
        this.schedule = function () {
            if (!self.live) { return; }
            clearTimeout(wait);
            wait = setTimeout(function () { self.layers.forEach(function (l) { self.update(l); }); }, 100);
        };
        map.on("moveend zoomend resize", function () { self.gen++; self.schedule(); });  // a new view
        map.on("overlayadd overlayremove", this.schedule);
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
        map.on("layeradd", function (e) {  // a band put on: its detail at once, when kept for this view
            var l = self.entry(e.layer);
            if (!l) { return; }
            var s = self.stored(l, self.view());
            if (s && s.img) { self.showImage(l, e.layer, s.a, s.box, s.res, s.img); self.around(l); }
            else if (s) { self.decode(l).then(function () { self.update(l); }); }
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

    // --- the details kept ------------------------------------------------------------------------
    Live.prototype.stored = function (l, v) {
        return l.store && this.fits(l.store, v, l) ? l.store : null;
    };
    Live.prototype.remember = function (l, a, box, res, img) {
        if (l.store && l.store.img) { this.forget(l); }
        l.store = {a: a, box: box, res: res, img: null};
        if (img) { this.mark(l, img); }
    };
    Live.prototype.mark = function (l, img) {  // decoded: counted, the oldest dropped past the budget
        if (!l.store || l.store.img === img) { return; }
        if (l.store.img) { this.forget(l); }
        l.store.img = img;
        l.store.bytes = 4 * (img.naturalWidth || 1) * (img.naturalHeight || 1);
        this.bytes += l.store.bytes;
        this.decoded.push(l);
        while (this.bytes > DECODED && this.decoded.length > 1) {
            var old = this.decoded[0];
            if (old.det && old.det.layer && window[old.v] && this.map.hasLayer(window[old.v])) {  // shown: keep
                this.decoded.push(this.decoded.shift());
                if (this.decoded[0] === old) { break; }
                continue;
            }
            this.forget(old);
        }
    };
    Live.prototype.forget = function (l) {  // its decoded image let go (its PNG kept)
        var i = this.decoded.indexOf(l);
        if (i >= 0) { this.decoded.splice(i, 1); }
        if (l.store && l.store.img) { this.bytes -= l.store.bytes || 0; l.store.img = null; }
    };
    Live.prototype.decode = function (l) {
        var self = this, s = l.store;
        if (!s) { return Promise.resolve(null); }
        if (s.img) { return Promise.resolve(s.img); }
        if (!s.decoding) {
            s.decoding = decoded(s.a).then(function (img) {
                s.decoding = null;
                if (l.store === s) { self.mark(l, img); }
                return img;
            });
        }
        return s.decoding;
    };
    // the bands around the one shown decoded beforehand (a slider's next steps draw at once)
    Live.prototype.around = function (l) {
        if (l.series === undefined) { return; }
        var self = this, v = this.view();
        this.layers.forEach(function (o) {
            if (o.series === l.series && o !== l && Math.abs(o.band - l.band) <= 3 && self.stored(o, v)) {
                self.decode(o);
            }
        });
    };
    // the other bands of a series asked for ahead, for the view, the nearest first, a few at a time:
    // a slider or the play button then finds them ready
    Live.prototype.ahead = function (l, rested) {
        if (l.series === undefined || l.kind !== "image") { return; }
        var self = this, gen = this.gen;
        if (!rested) {  // once the view has rested a moment: not for every step of a zoom
            setTimeout(function () { if (self.gen === gen) { self.ahead(l, true); } }, 300);
            return;
        }
        var v = this.view(), p = this.plan(l, v);
        if (!p || this.asking === gen) { return; }
        var todo = this.layers.filter(function (o) { return o.series === l.series && o !== l && !self.stored(o, v); });
        todo.sort(function (a, b) { return Math.abs(a.band - l.band) - Math.abs(b.band - l.band); });
        if (!todo.length) { return; }
        this.asking = gen;
        var next = function () {
            if (self.gen !== gen || !todo.length) { if (self.asking === gen) { self.asking = null; } return; }
            var part = todo.splice(0, AHEAD);
            ask({op: "details", map: self.id, keys: part.map(function (o) { return o.key; }), bbox: p.box, w: p.w, h: p.h})
                .then(function (r) {
                    if (self.gen !== gen) { if (self.asking === gen) { self.asking = null; } return; }
                    ((r && r.answers) || []).forEach(function (a, i) {
                        if (a && a.ok) { self.remember(part[i], a, p.box, p.res, null); }
                    });
                    self.around(l);
                    next();
                });
        };
        next();
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
        if (l.det && this.fits(l.det, v, l)) { if (!target) { this.ahead(l); } return; }  // still good
        var s = this.stored(l, v);
        if (s && !target) {  // kept for this view: drawn from it
            this.decode(l).then(function (img) {
                if (img && map.hasLayer(base) && !(l.det && self.fits(l.det, self.view(), l))) {
                    self.showImage(l, base, s.a, s.box, s.res, img);
                    self.around(l);
                }
            });
            this.ahead(l);
            return;
        }
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
                if (l.kind === "gl") { self.showGL(l, base, a, p.box, p.res); return; }
                self.remember(l, a, p.box, p.res, img);
                self.showImage(l, base, a, p.box, p.res, img);
                self.ahead(l);
            };
            if (l.kind === "gl") { show(null); } else { decoded(a).then(show); }
        });
    };

    // the detail of a layer not on the map yet (the next band played), for the view: the band is
    // put on with it, never coarse first. Resolves when it is ready, at most after ``wait`` ms.
    Live.prototype.prepare = function (l, wait) {
        if (!this.live || l.kind !== "image") { return Promise.resolve(); }
        var self = this, v = this.view(), p = this.plan(l, v);
        if (!p) { return Promise.resolve(); }
        if (l.det && this.fits(l.det, v, l)) { return Promise.resolve(); }
        var done;
        if (this.stored(l, v)) {
            done = this.decode(l);
        } else {
            done = ask({op: "detail", map: this.id, key: l.key, bbox: p.box, w: p.w, h: p.h}).then(function (a) {
                if (!a || !a.ok) { return; }
                return decoded(a).then(function (img) { self.remember(l, a, p.box, p.res, img); });
            });
        }
        return wait ? Promise.race([done, new Promise(function (r) { setTimeout(r, wait); })]) : done;
    };

    Live.prototype.showImage = function (l, base, a, box, px, img) {
        var map = this.map, old = l.det;
        if (old && old.a === a && old.layer && map.hasLayer(old.layer)) { return; }  // already shown
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
        }
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
