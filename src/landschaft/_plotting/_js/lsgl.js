/* landschaft (GPL-3.0-or-later): large vector layers drawn with WebGL on a Leaflet map.
 *
 * The features come from Python in Web Mercator, simplified, quantized to a grid of q metres from
 * an origin and cut into triangles (earcut, in C++); integer arrays as zigzag deltas in byte
 * planes, all deflated together. One canvas per layer, a Leaflet renderer: it follows the map's
 * pans and zoom animations as Leaflet's own canvas does. The feature under the cursor is found
 * by drawing the features' numbers as colours around that pixel (GPU picking).
 */
(function () {
    "use strict";
    if (L.LsGL) { return; }

    function bytes(b64) {
        var bin = atob(b64), n = bin.length, a = new Uint8Array(n);
        for (var i = 0; i < n; i++) { a[i] = bin.charCodeAt(i); }
        return a;
    }
    function inflate1(u8) {
        var s = new Blob([u8]).stream().pipeThrough(new DecompressionStream("deflate"));
        return new Response(s).arrayBuffer().then(function (b) { return new Uint8Array(b); });
    }
    function inflate(all, pieces) {  // pieces deflated one by one: [offset, length] in all
        if (typeof DecompressionStream === "undefined") {
            return Promise.reject(new Error("this browser cannot inflate data (DecompressionStream)"));
        }
        return Promise.all(pieces.map(function (p) { return inflate1(all.subarray(p[0], p[0] + p[1])); }))
            .then(function (parts) {
                if (parts.length === 1) { return parts[0]; }
                var n = 0, at = 0;
                parts.forEach(function (x) { n += x.length; });
                var out = new Uint8Array(n);
                parts.forEach(function (x) { out.set(x, at); at += x.length; });
                return out;
            });
    }
    function planes(u8, off, n) {  // n uint32 stored as four planes of bytes
        var out = new Uint32Array(n), p1 = off + n, p2 = off + 2 * n, p3 = off + 3 * n;
        for (var i = 0; i < n; i++) {
            out[i] = (u8[off + i] | (u8[p1 + i] << 8) | (u8[p2 + i] << 16) | (u8[p3 + i] << 24)) >>> 0;
        }
        return out;
    }
    function undelta(u) {  // zigzag deltas: the values
        var out = new Int32Array(u.length), acc = 0;
        for (var i = 0; i < u.length; i++) { var v = u[i]; acc = (acc + ((v >>> 1) ^ -(v & 1))) | 0; out[i] = acc; }
        return out;
    }

    // the arrays of a layer, as Python wrote them (see _gl.py), deflated in pieces: the
    // [offset, length] of its pieces in the payload, its number of values, whether deltas
    function decode(spec, b64) {
        var all = bytes(b64), a = {props: null};
        var jobs = Object.keys(spec.arrays).map(function (k) {
            var e = spec.arrays[k];
            return inflate(all, e[0]).then(function (u8) {
                var v = planes(u8, 0, e[1]);
                a[k] = e[2] ? undelta(v) : v;
            });
        });
        var p = spec.props;
        if (p) {
            jobs.push(inflate(all, p).then(function (u8) {
                a.props = JSON.parse(new TextDecoder().decode(u8));
            }));
        }
        return Promise.all(jobs).then(function () { return build(spec, a); });
    }

    // GPU buffers: positions (metres from the origin), colours, feature numbers as colours (picking)
    function build(spec, a) {
        var q = spec.q, pal = spec.palette, fc = a.fcol;
        var out = {props: a.props, fcol: fc};
        function vertices(xy, feat) {  // feat: the feature of each vertex
            var n = feat.length, pos = new Float32Array(2 * n), col = new Uint8Array(4 * n), id = new Uint8Array(4 * n);
            for (var i = 0; i < n; i++) {
                pos[2 * i] = xy[i] * q; pos[2 * i + 1] = xy[n + i] * q;
                var f = feat[i], c = pal[fc[f]], k = f + 1;
                col[4 * i] = c[0]; col[4 * i + 1] = c[1]; col[4 * i + 2] = c[2]; col[4 * i + 3] = 255;
                id[4 * i] = k & 255; id[4 * i + 1] = (k >> 8) & 255; id[4 * i + 2] = (k >> 16) & 255; id[4 * i + 3] = 255;
            }
            return {pos: pos, col: col, id: id};
        }
        if (a.pxy) {  // polygons: rings, the rings of each polygon, its feature and its triangles
            var nv = a.pxy.length / 2, rl = a.pring, pr = a.prings, pf = a.pfeat, pt = a.ptri, li = a.pidx;
            var feat = new Uint32Array(nv), tri = new Uint32Array(li.length), edges = [], r = 0, v = 0, t = 0, f = 0;
            var nedge = 0;
            for (var j = 0; j < rl.length; j++) { nedge += Math.max(0, rl[j] - 1); }
            var ed = new Uint32Array(2 * nedge), e = 0;
            for (var p = 0; p < pr.length; p++) {
                f += pf[p];
                var v0 = v;
                for (var k = 0; k < pr[p]; k++, r++) {
                    for (var i = 0; i < rl[r]; i++) {
                        feat[v + i] = f;
                        if (i > 0) { ed[e++] = v + i - 1; ed[e++] = v + i; }
                    }
                    v += rl[r];
                }
                for (var m = 0; m < pt[p]; m++, t++) { tri[t] = li[t] + v0; }
            }
            out.poly = vertices(a.pxy, feat);
            out.poly.tri = tri; out.poly.edges = ed;
        }
        if (a.lxy) {  // lines: their lengths and features
            var nl = a.lxy.length / 2, ll = a.llen, lf = a.lfeat;
            var featl = new Uint32Array(nl), segs = 0;
            for (var j2 = 0; j2 < ll.length; j2++) { segs += Math.max(0, ll[j2] - 1); }
            var sg = new Uint32Array(2 * segs), s = 0, w = 0, g = 0;
            for (var l = 0; l < ll.length; l++) {
                g += lf[l];
                for (var i2 = 0; i2 < ll[l]; i2++) {
                    featl[w + i2] = g;
                    if (i2 > 0) { sg[s++] = w + i2 - 1; sg[s++] = w + i2; }
                }
                w += ll[l];
            }
            out.line = vertices(a.lxy, featl);
            out.line.segs = sg;
        }
        if (a.txy) {  // points
            var nt = a.txy.length / 2, tf = a.tfeat, featt = new Uint32Array(nt), h = 0;
            for (var i3 = 0; i3 < nt; i3++) { h += tf[i3]; featt[i3] = h; }
            out.point = vertices(a.txy, featt);
        }
        return out;
    }

    var VS = "attribute vec2 a_pos; attribute vec4 a_col; attribute vec4 a_id;" +
        "uniform vec2 u_a; uniform vec2 u_b; uniform float u_size; uniform float u_pick;" +
        "varying vec4 v_col;" +
        "void main() { gl_Position = vec4(a_pos * u_a + u_b, 0.0, 1.0); gl_PointSize = u_size;" +
        "  v_col = u_pick > 0.5 ? a_id : a_col; }";
    var FS = "#ifdef GL_FRAGMENT_PRECISION_HIGH\nprecision highp float;\n#else\nprecision mediump float;\n#endif\n" +
        "varying vec4 v_col; uniform float u_alpha; uniform float u_round;" +
        "uniform float u_pick;" +
        "void main() { if (u_round > 0.5) { vec2 d = gl_PointCoord - 0.5; if (dot(d, d) > 0.25) { discard; } }" +
        "  if (u_pick > 0.5) { gl_FragColor = v_col; } else { gl_FragColor = vec4(v_col.rgb * u_alpha, u_alpha); } }";

    function program(gl) {
        function shader(type, src) {
            var s = gl.createShader(type);
            gl.shaderSource(s, src); gl.compileShader(s);
            if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) { throw new Error(gl.getShaderInfoLog(s)); }
            return s;
        }
        var p = gl.createProgram();
        gl.attachShader(p, shader(gl.VERTEX_SHADER, VS)); gl.attachShader(p, shader(gl.FRAGMENT_SHADER, FS));
        gl.linkProgram(p);
        if (!gl.getProgramParameter(p, gl.LINK_STATUS)) { throw new Error(gl.getProgramInfoLog(p)); }
        var u = {};
        ["u_a", "u_b", "u_size", "u_pick", "u_alpha", "u_round"].forEach(function (n) { u[n] = gl.getUniformLocation(p, n); });
        return {p: p, u: u, pos: gl.getAttribLocation(p, "a_pos"), col: gl.getAttribLocation(p, "a_col"),
                id: gl.getAttribLocation(p, "a_id")};
    }

    var R = 6378137;

    L.LsGL = L.Renderer.extend({
        options: {padding: 0.3},

        initialize: function (spec, data, options) {
            L.Renderer.prototype.initialize.call(this, options);
            this._spec = spec;
            this._fill = spec.fill; this._line = spec.line;
            var self = this;
            this._ready = decode(spec, data).then(function (d) {
                self._data = d;
                if (self._gl) { self._upload(); self._draw(); }
            }).catch(function (e) {
                self._failed = String(e && e.message || e);
                if (window.console) { console.error("landschaft: the layer " + spec.name + " is not drawn: " + self._failed); }
            });
            (window.LS_GL = window.LS_GL || []).push(this);
        },

        _initContainer: function () {
            var c = this._container = document.createElement("canvas");
            c.style.pointerEvents = "none";
            c.className = "ls-gl";
            var gl = c.getContext("webgl", {premultipliedAlpha: true, antialias: true, alpha: true}) ||
                     c.getContext("experimental-webgl");
            if (!gl) { this._failed = "no WebGL"; return; }
            this._gl = gl;
            gl.getExtension("OES_element_index_uint");
            this._prog = program(gl);
            var self = this;
            c.addEventListener("webglcontextlost", function (e) { e.preventDefault(); self._buf = null; });
            c.addEventListener("webglcontextrestored", function () {
                self._prog = program(gl); self._fb = null;
                if (self._data) { self._upload(); self._draw(); }
            });
            if (this._data) { this._upload(); }
        },

        _destroyContainer: function () {
            L.DomUtil.remove(this._container);
        },

        _updatePaths: function () {},

        _update: function () {
            if (this._map._animatingZoom && this._bounds) { return; }
            L.Renderer.prototype._update.call(this);
            var b = this._bounds, size = b.getSize(), m = window.devicePixelRatio || 1, c = this._container;
            L.DomUtil.setPosition(c, b.min);
            c.width = Math.max(1, Math.round(m * size.x)); c.height = Math.max(1, Math.round(m * size.y));
            c.style.width = size.x + "px"; c.style.height = size.y + "px";
            this._draw();
            this.fire("update");
        },

        _upload: function () {
            var gl = this._gl, d = this._data, buf = {};
            function array(target, data) {
                var b = gl.createBuffer();
                gl.bindBuffer(target, b); gl.bufferData(target, data, gl.STATIC_DRAW);
                return b;
            }
            ["poly", "line", "point"].forEach(function (k) {
                var x = d[k];
                if (!x) { return; }
                var o = {pos: array(gl.ARRAY_BUFFER, x.pos), col: array(gl.ARRAY_BUFFER, x.col),
                         id: array(gl.ARRAY_BUFFER, x.id), n: x.pos.length / 2};
                if (x.tri) { o.tri = array(gl.ELEMENT_ARRAY_BUFFER, x.tri); o.ntri = x.tri.length; }
                if (x.edges) { o.edges = array(gl.ELEMENT_ARRAY_BUFFER, x.edges); o.nedges = x.edges.length; }
                if (x.segs) { o.segs = array(gl.ELEMENT_ARRAY_BUFFER, x.segs); o.nsegs = x.segs.length; }
                buf[k] = o;
            });
            this._buf = buf;
        },

        // clip space = position * a + b, the transform worked out in double precision
        _view: function () {
            var map = this._map, z = this._zoom, b = this._bounds, size = b.getSize();
            var scale = 256 * Math.pow(2, z), s = scale * 0.5 / (Math.PI * R), po = map.getPixelOrigin();
            var o = this._spec.origin;
            var cx = s * o[0] + scale * 0.5 - po.x - b.min.x, cy = -s * o[1] + scale * 0.5 - po.y - b.min.y;
            return {a: [2 * s / size.x, 2 * s / size.y], b: [2 * cx / size.x - 1, 1 - 2 * cy / size.y]};
        },

        _pass: function (pick) {
            var gl = this._gl, P = this._prog, B = this._buf, v = this._view(), m = window.devicePixelRatio || 1;
            gl.useProgram(P.p);
            gl.uniform2f(P.u.u_a, v.a[0], v.a[1]); gl.uniform2f(P.u.u_b, v.b[0], v.b[1]);
            gl.uniform1f(P.u.u_pick, pick ? 1 : 0);
            function bind(o) {
                gl.bindBuffer(gl.ARRAY_BUFFER, o.pos); gl.enableVertexAttribArray(P.pos);
                gl.vertexAttribPointer(P.pos, 2, gl.FLOAT, false, 0, 0);
                gl.bindBuffer(gl.ARRAY_BUFFER, o.col); gl.enableVertexAttribArray(P.col);
                gl.vertexAttribPointer(P.col, 4, gl.UNSIGNED_BYTE, true, 0, 0);
                gl.bindBuffer(gl.ARRAY_BUFFER, o.id); gl.enableVertexAttribArray(P.id);
                gl.vertexAttribPointer(P.id, 4, gl.UNSIGNED_BYTE, true, 0, 0);
            }
            gl.uniform1f(P.u.u_round, 0);
            gl.uniform1f(P.u.u_size, 1);
            if (B.poly) {
                bind(B.poly);
                gl.uniform1f(P.u.u_alpha, this._fill);
                gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, B.poly.tri);
                gl.drawElements(gl.TRIANGLES, B.poly.ntri, gl.UNSIGNED_INT, 0);
                if (!pick && B.poly.nedges) {
                    gl.uniform1f(P.u.u_alpha, this._line);
                    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, B.poly.edges);
                    gl.drawElements(gl.LINES, B.poly.nedges, gl.UNSIGNED_INT, 0);
                }
            }
            if (B.line) {
                bind(B.line);
                gl.uniform1f(P.u.u_alpha, this._line);
                gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, B.line.segs);
                gl.drawElements(gl.LINES, B.line.nsegs, gl.UNSIGNED_INT, 0);
            }
            if (B.point) {
                bind(B.point);
                gl.uniform1f(P.u.u_round, 1);
                gl.uniform1f(P.u.u_size, 8 * m);
                gl.uniform1f(P.u.u_alpha, Math.min(1, this._line));
                gl.drawArrays(gl.POINTS, 0, B.point.n);
            }
        },

        _draw: function () {
            var gl = this._gl;
            if (!gl || !this._buf || !this._bounds || gl.isContextLost()) { return; }
            gl.bindFramebuffer(gl.FRAMEBUFFER, null);
            gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
            gl.disable(gl.SCISSOR_TEST);
            gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
            if (this._suppress) { return; }  // a layer of the part in view, in detail, drawn instead
            gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
            this._pass(false);
        },

        // the feature drawn at a point of the map (the nearest within a few pixels), or -1
        pick: function (latlng) {
            var gl = this._gl, map = this._map;
            if (!gl || !this._buf || !map || map._animatingZoom || this._suppress || gl.isContextLost()) { return -1; }
            var W = gl.drawingBufferWidth, H = gl.drawingBufferHeight, m = window.devicePixelRatio || 1;
            var p = map.latLngToLayerPoint(latlng).subtract(this._bounds.min).multiplyBy(m);
            var x = Math.round(p.x), y = H - 1 - Math.round(p.y), r = Math.max(2, Math.round(3 * m));
            if (x < 0 || y < 0 || x >= W || y >= H) { return -1; }
            if (!this._fb || this._fbw !== W || this._fbh !== H) {
                if (this._fb) { gl.deleteFramebuffer(this._fb); gl.deleteTexture(this._tex); }
                this._fb = gl.createFramebuffer(); this._tex = gl.createTexture();  // 8 bits a channel
                gl.bindTexture(gl.TEXTURE_2D, this._tex);
                gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, W, H, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
                gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
                gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
                gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
                gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
                gl.bindFramebuffer(gl.FRAMEBUFFER, this._fb);
                gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this._tex, 0);
                this._fbw = W; this._fbh = H;
            }
            gl.bindFramebuffer(gl.FRAMEBUFFER, this._fb);
            gl.viewport(0, 0, W, H);
            var x0 = Math.max(0, x - r), y0 = Math.max(0, y - r), w = Math.min(W, x + r + 1) - x0, h = Math.min(H, y + r + 1) - y0;
            gl.enable(gl.SCISSOR_TEST); gl.scissor(x0, y0, w, h);
            gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
            gl.disable(gl.BLEND);
            this._pass(true);
            var px = new Uint8Array(4 * w * h);
            gl.readPixels(x0, y0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, px);
            gl.disable(gl.SCISSOR_TEST);
            gl.bindFramebuffer(gl.FRAMEBUFFER, null);
            var best = -1, dist = Infinity;
            for (var j = 0; j < h; j++) {
                for (var i = 0; i < w; i++) {
                    var o = 4 * (j * w + i), k = px[o] | (px[o + 1] << 8) | (px[o + 2] << 16);
                    if (!k) { continue; }
                    var d = (x0 + i - x) * (x0 + i - x) + (y0 + j - y) * (y0 + j - y);
                    if (d < dist) { dist = d; best = k - 1; }
                }
            }
            return best;
        },

        // the value of the colouring column of a feature: held, else read back from its colour
        // (its class, or its place on the ramp, as for images)
        _value: function (f) {
            var s = this._spec, d = this._data, props = d && d.props, u = s.lookup;
            if (s.column && props && props[s.column]) {
                var v = props[s.column][f];
                return v === null ? "no value" : v;
            }
            if (!u || !d) { return "feature " + (f + 1); }
            var c = d.fcol[f];
            if (c >= s.palette.length - 1) { return "no value"; }
            if (u.labels) { return u.labels[c]; }
            var fmt = function (x) {
                var a = Math.abs(x);
                return (a !== 0 && (a >= 1e6 || a < 1e-3)) ? x.toExponential(3) : String(+x.toPrecision(4));
            };
            if (c <= 0 && u.under) { return "\u2264 " + fmt(u.vmin); }
            if (c >= 254 && u.over) { return "\u2265 " + fmt(u.vmax); }
            var x = c / 254;
            return "\u2248 " + fmt(u.log ? Math.pow(10, Math.log10(u.vmin) + x * (Math.log10(u.vmax) - Math.log10(u.vmin)))
                                           : u.vmin + x * (u.vmax - u.vmin));
        },

        // [label, value] rows of the feature at a point: the colouring column (hover), or all the
        // columns of the tooltip (full: a click)
        rowsAt: function (latlng, full) {
            var f = this.pick(latlng), s = this._spec;
            if (f < 0) { return []; }
            var props = this._data && this._data.props, out = [];
            if (!full || !(s.fields || []).length) { return [[s.name, this._value(f)]]; }
            out.push([s.name, ""]);
            (s.fields || []).forEach(function (k) {
                var v = props && props[k] ? props[k][f] : null;
                out.push([k, v === null ? "" : v]);
            });
            return out;
        },

        setOpacity: function (o) {
            this._fill = o * 0.6; this._line = Math.min(1, o + 0.2);
            this._draw();
            return this;
        }
    });
})();
