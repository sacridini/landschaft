// SPDX-License-Identifier: GPL-3.0-or-later
//
// Python bindings of movement ecology (landschaft._core.move_*).

#include "move/python.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "move/amt.hpp"
#include "move/distr.hpp"
#include "move/geo.hpp"
#include "move/rnmath.hpp"
#include "move/track.hpp"
#include "nlm/rng.hpp"

namespace py = pybind11;

namespace {

using F64 = py::array_t<double, py::array::c_style | py::array::forcecast>;
using I64 = py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>;

std::int64_t check(const F64& x, const F64& y, const I64& track) {
    if (x.ndim() != 1 || y.size() != x.size() || track.size() != x.size())
        throw std::invalid_argument("x, y and track: 1-D arrays of the same length");
    return static_cast<std::int64_t>(x.size());
}

}  // namespace

namespace move {

void register_python(py::module_& m) {
    m.def(
        "move_distances",
        [](const F64& x, const F64& y, const I64& track, bool lonlat, int threads) {
            const std::int64_t n = check(x, y, track);
            F64 out(n);
            {
                py::gil_scoped_release release;
                distances(x.data(), y.data(), track.data(), n, lonlat ? Coords::LonLat : Coords::Planar, threads,
                          out.mutable_data());
            }
            return out;
        },
        py::arg("x"), py::arg("y"), py::arg("track"), py::arg("lonlat"), py::arg("threads") = 1,
        "Distance from each location to the next of its track (move2's mt_distance; NaN at the last).");

    m.def(
        "move_azimuths",
        [](const F64& x, const F64& y, const I64& track, bool lonlat, double a, double inv_flattening,
           int threads) {
            const std::int64_t n = check(x, y, track);
            F64 out(n);
            {
                py::gil_scoped_release release;
                azimuths(x.data(), y.data(), track.data(), n, lonlat ? Coords::LonLat : Coords::Planar, a,
                         inv_flattening, threads, out.mutable_data());
            }
            return out;
        },
        py::arg("x"), py::arg("y"), py::arg("track"), py::arg("lonlat"), py::arg("a") = 6378137.0,
        py::arg("inv_flattening") = 298.257223563, py::arg("threads") = 1,
        "Azimuth (radians) of the step from each location to the next (move2's mt_azimuth).");

    m.def(
        "move_turn_angles",
        [](const F64& azimuth, int threads) {
            const auto n = static_cast<std::int64_t>(azimuth.size());
            F64 out(n);
            {
                py::gil_scoped_release release;
                turn_angles(azimuth.data(), n, threads, out.mutable_data());
            }
            return out;
        },
        py::arg("azimuth"), py::arg("threads") = 1, "Turn angles from the azimuths (move2's mt_turnangle).");

    m.def(
        "move_interpolate",
        [](const F64& x0, const F64& y0, const F64& x1, const F64& y1, const F64& fraction, bool lonlat) {
            const auto n = static_cast<std::int64_t>(x0.size());
            if (y0.size() != n || x1.size() != n || y1.size() != n || fraction.size() != n)
                throw std::invalid_argument("x0, y0, x1, y1 and fraction: the same length");
            F64 x(n), y(n);
            double* px = x.mutable_data();
            double* py_ = y.mutable_data();
            for (std::int64_t i = 0; i < n; ++i) {
                if (lonlat)
                    s2_interpolate(x0.data()[i], y0.data()[i], x1.data()[i], y1.data()[i], fraction.data()[i], px[i],
                                   py_[i]);
                else
                    planar_sample(x0.data()[i], y0.data()[i], x1.data()[i], y1.data()[i], fraction.data()[i], px[i],
                                  py_[i]);
            }
            return py::make_tuple(x, y);
        },
        py::arg("x0"), py::arg("y0"), py::arg("x1"), py::arg("y1"), py::arg("fraction"), py::arg("lonlat"),
        "Points at a fraction of the steps: s2's interpolation in lon / lat, sf's st_line_sample otherwise.");

    m.def(
        "move_s2_centroid",
        [](const F64& lon, const F64& lat) {
            if (lat.size() != lon.size()) throw std::invalid_argument("lon and lat: the same length");
            double x = 0, y = 0;
            s2_centroid(lon.data(), lat.data(), static_cast<long long>(lon.size()), x, y);
            return py::make_tuple(x, y);
        },
        py::arg("lon"), py::arg("lat"), "s2's centroid of points (degrees).");

    m.def("move_r_fmod", &r_fmod, py::arg("x1"), py::arg("x2"), "R's x %% y for doubles.");
    m.def("move_r_fdiv", &r_fdiv, py::arg("x1"), py::arg("x2"), "R's x %/% y for doubles.");

    m.def(
        "move_brownian",
        [](const std::vector<F64>& times, const std::vector<F64>& sigmas, const F64& starts, std::int32_t seed) {
            if (times.size() != sigmas.size() || starts.ndim() != 2 || starts.shape(0) != py::ssize_t(times.size()) ||
                starts.shape(1) != 2)
                throw std::invalid_argument("a time vector, a sigma and a start (x, y) per track");
            nlm::RRng rng(seed);
            py::list out;
            for (std::size_t k = 0; k < times.size(); ++k) {
                const auto n = static_cast<std::int64_t>(times[k].size());
                F64 x(n), y(n);
                brownian(&rng, times[k].data(), n, sigmas[k].data(), static_cast<std::int64_t>(sigmas[k].size()),
                         starts.at(k, 0), starts.at(k, 1), x.mutable_data(), y.mutable_data());
                out.append(py::make_tuple(x, y));
            }
            return out;
        },
        py::arg("times"), py::arg("sigmas"), py::arg("starts"), py::arg("seed"),
        "mt_sim_brownian_motion: the tracks one after the other from set.seed(seed).");

    m.def(
        "move_r_fmod_array",
        [](const F64& x, double y) {
            F64 out(x.size());
            for (py::ssize_t i = 0; i < x.size(); ++i) out.mutable_data()[i] = r_fmod(x.data()[i], y);
            return out;
        },
        py::arg("x"), py::arg("y"), "R's x %% y for a vector and a number.");

    m.def(
        "move_amt_steps",
        [](const F64& x, const F64& y, const I64& group, bool lonlat) {
            const std::int64_t n = check(x, y, group);
            F64 sl(n), dir(n), ta(n);
            {
                py::gil_scoped_release release;
                amt_steps(x.data(), y.data(), group.data(), n, lonlat, sl.mutable_data(), dir.mutable_data(),
                          ta.mutable_data());
            }
            return py::make_tuple(sl, dir, ta);
        },
        py::arg("x"), py::arg("y"), py::arg("group"), py::arg("lonlat"),
        "amt's step lengths, directions and turn angles (NaN at the last location of each group).");

    m.def(
        "move_amt_tac", [](const F64& ta) { return amt_tac(ta.data(), static_cast<std::int64_t>(ta.size())); },
        py::arg("ta"), "amt's tac of a track's steps' turn angles.");
    m.def(
        "move_amt_sinuosity",
        [](const F64& sl, const F64& ta) {
            if (sl.size() != ta.size() || sl.size() < 2) throw std::invalid_argument("sl and ta: two steps or more");
            return amt_sinuosity(sl.data(), ta.data(), static_cast<std::int64_t>(sl.size()));
        },
        py::arg("sl"), py::arg("ta"), "Benhamou's sinuosity (amt's, the angles in radians).");
    m.def(
        "move_amt_msd",
        [](const F64& x, const F64& y) {
            if (x.size() != y.size() || x.size() < 1) throw std::invalid_argument("x and y: the same length");
            return amt_msd(x.data(), y.data(), static_cast<std::int64_t>(x.size()));
        },
        py::arg("x"), py::arg("y"), "amt's msd: mean squared distance to the mean location.");
    m.def(
        "move_r_sum", [](const F64& x) { return r_sum_of(x.data(), static_cast<std::int64_t>(x.size())); },
        py::arg("x"), "R's sum() of doubles (long double accumulation).");
    m.def(
        "move_r_mean", [](const F64& x) { return r_mean_of(x.data(), static_cast<std::int64_t>(x.size())); },
        py::arg("x"), "R's mean() of doubles.");

    m.def(
        "move_dgamma",
        [](const F64& x, double shape, double scale, bool give_log) {
            F64 out(x.size());
            for (py::ssize_t i = 0; i < x.size(); ++i) out.mutable_data()[i] = r_dgamma(x.data()[i], shape, scale, give_log);
            return out;
        },
        py::arg("x"), py::arg("shape"), py::arg("scale"), py::arg("log") = false, "R's dgamma.");
    m.def(
        "move_lgammafn",
        [](const F64& x) {
            F64 out(x.size());
            for (py::ssize_t i = 0; i < x.size(); ++i) out.mutable_data()[i] = r_lgammafn(x.data()[i]);
            return out;
        },
        py::arg("x"), "R's lgamma.");

    m.def(
        "move_fit_gamma",
        [](const F64& x) {
            GammaFit f;
            {
                py::gil_scoped_release release;
                f = fit_gamma(x.data(), static_cast<std::int64_t>(x.size()));
            }
            py::dict d;
            d["shape"] = f.shape;
            d["scale"] = f.scale;
            d["vcov"] = f.vcov;
            d["fail"] = f.fail;
            d["fncount"] = f.fncount;
            d["loglik"] = f.loglik;
            d["zero_replacement"] = f.zeros_replaced ? py::object(py::float_(f.zero_replacement)) : py::none();
            return d;
        },
        py::arg("x"), "amt's fit_distr(x, \"gamma\") (fitdistrplus by Nelder-Mead).");
    m.def(
        "move_fit_vonmises",
        [](const F64& x) {
            const VonMisesFit f = fit_vonmises(x.data(), static_cast<std::int64_t>(x.size()));
            py::dict d;
            d["mu"] = f.mu;
            d["kappa"] = f.kappa;
            d["se_mu"] = f.se_mu;
            d["se_kappa"] = f.se_kappa;
            return d;
        },
        py::arg("x"), "circular's mle.vonmises (amt's fit_distr(x, \"vonmises\")).");
    m.def(
        "move_fit_exp", [](const F64& x) { return fit_exp(x.data(), static_cast<std::int64_t>(x.size())); },
        py::arg("x"), "amt's fit_distr(x, \"exp\"): the rate (fitdistrplus by BFGS).");
    m.def(
        "move_fit_lnorm",
        [](const F64& x) {
            const auto p = fit_lnorm(x.data(), static_cast<std::int64_t>(x.size()));
            return py::make_tuple(p[0], p[1]);
        },
        py::arg("x"), "amt's fit_distr(x, \"lnorm\"): (meanlog, sdlog) (fitdistrplus by Nelder-Mead).");

    m.def(
        "move_random_numbers",
        [](std::int32_t seed, const std::string& name, const std::vector<double>& params, std::int64_t n) {
            nlm::RRng rng(seed);
            F64 out(n);
            random_numbers(rng, name.c_str(), params, n, out.mutable_data());
            return out;
        },
        py::arg("seed"), py::arg("name"), py::arg("params"), py::arg("n"),
        "amt's random_numbers(distr, n) after set.seed(seed).");

    m.def(
        "move_amt_resample",
        [](const F64& t, const I64& group, double rate, double tolerance, std::int64_t start) {
            if (group.size() != t.size()) throw std::invalid_argument("t and group: the same length");
            I64 out(t.size());
            amt_resample(t.data(), group.data(), static_cast<std::int64_t>(t.size()), rate, tolerance, start,
                         out.mutable_data());
            return out;
        },
        py::arg("t"), py::arg("group"), py::arg("rate"), py::arg("tolerance"), py::arg("start"),
        "amt's track_resample bursts per group (-1: left out).");

    m.def(
        "move_amt_controls",
        [](const F64& x1, const F64& y1, const F64& x0, const F64& y0, const F64& angle, const F64& sl,
           const F64& ta) {
            const auto n = static_cast<std::int64_t>(x1.size());
            for (const F64* a : {&y1, &x0, &y0, &angle, &sl, &ta})
                if (a->size() != x1.size()) throw std::invalid_argument("arrays of the same length");
            F64 x2(n), y2(n), tao(n);
            amt_controls(x1.data(), y1.data(), x0.data(), y0.data(), angle.data(), sl.data(), ta.data(), n,
                         x2.mutable_data(), y2.mutable_data(), tao.mutable_data());
            return py::make_tuple(x2, y2, tao);
        },
        py::arg("x1"), py::arg("y1"), py::arg("x0"), py::arg("y0"), py::arg("angle"), py::arg("sl"), py::arg("ta"),
        "amt's control steps: their ends and turn angles.");

    m.def(
        "move_sun_times",
        [](const F64& lon, const F64& lat, const F64& noon) {
            const auto n = static_cast<std::int64_t>(lon.size());
            if (lat.size() != lon.size() || noon.size() != lon.size()) throw std::invalid_argument("same lengths");
            F64 a(n), b(n), c(n), d(n);
            for (std::int64_t i = 0; i < n; ++i)
                sun_times(lon.data()[i], lat.data()[i], noon.data()[i], a.mutable_data() + i, b.mutable_data() + i,
                          c.mutable_data() + i, d.mutable_data() + i);
            return py::make_tuple(a, b, c, d);
        },
        py::arg("lon"), py::arg("lat"), py::arg("noon"), "suncalc's sunrise, sunset, dawn and dusk (seconds).");

    m.def(
        "move_random_step_draws",
        [](std::int32_t seed, const py::list& calls, std::int64_t n_random, std::int64_t n_control) {
            // set.seed(seed), then for each call of random_steps in turn: rand_sl <-
            // random_numbers(sl, n_random); rand_ta <- random_numbers(ta, n_random); for each
            // step, sample.int(n_random, n_control, TRUE) of each
            if (n_random < 1 || n_control < 1) throw std::invalid_argument("n_random and n_control: >= 1");
            nlm::RRng rng(seed);
            py::list out;
            std::vector<double> rsl(static_cast<std::size_t>(n_random)), rta(static_cast<std::size_t>(n_random));
            for (const auto& item : calls) {
                const auto call = item.cast<py::tuple>();
                const auto sl_name = call[0].cast<std::string>();
                const auto sl_params = call[1].cast<std::vector<double>>();
                const auto ta_name = call[2].cast<std::string>();
                const auto ta_params = call[3].cast<std::vector<double>>();
                const auto n_steps = call[4].cast<std::int64_t>();
                // sample.int(n, size) calls before the draws: the subsample fitdistrplus keeps
                // (keepdata = FALSE) when the step lengths' distribution is fitted then
                for (const auto& burn : call[5].cast<std::vector<std::pair<std::int64_t, std::int64_t>>>())
                    rng.sample_int(burn.first, burn.second, false);
                random_numbers(rng, sl_name.c_str(), sl_params, n_random, rsl.data());
                random_numbers(rng, ta_name.c_str(), ta_params, n_random, rta.data());
                py::array_t<double> sl({n_steps, n_control}), ta({n_steps, n_control});
                double* ps = sl.mutable_data();
                double* pt = ta.mutable_data();
                for (std::int64_t k = 0; k < n_steps; ++k) {
                    const auto is = rng.sample_int(n_random, n_control, true);
                    const auto it = rng.sample_int(n_random, n_control, true);
                    for (std::int64_t j = 0; j < n_control; ++j) {
                        ps[k * n_control + j] = rsl[static_cast<std::size_t>(is[static_cast<std::size_t>(j)] - 1)];
                        pt[k * n_control + j] = rta[static_cast<std::size_t>(it[static_cast<std::size_t>(j)] - 1)];
                    }
                }
                out.append(py::make_tuple(sl, ta));
            }
            return out;
        },
        py::arg("seed"), py::arg("calls"), py::arg("n_random"), py::arg("n_control"),
        "amt's random_steps draws for calls (sl name, sl params, ta name, ta params, n steps, [(n, size)...]) in turn.");
}

}  // namespace move
