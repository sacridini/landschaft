# SPDX-License-Identifier: GPL-3.0-or-later
"""R's stats::kmeans (landschaft._core.sdm_kmeans) against R 4.4.2, bit for bit.

The references (CASES, AS_CHARACTER) come from tools/kmeans_reference.R; the
data sets are rebuilt here with R's generator (_core.RRandom), and each case
checks the clusters, centres, sums of squares, warnings and the generator's
state afterwards (the next runif(1)).
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from landschaft import _core

QTRAN = "1200100001202112122002221020221120222201022100012210221212220021201221112221201121222221212012222111"


def make(name: str) -> np.ndarray:
    """The data sets of tools/kmeans_reference.R (rows = observations)."""
    R = _core.RRandom
    if name == "coords":
        r = R(11)
        return np.column_stack([r.runif(60, -60, -40), r.runif(60, -20, 0)])
    if name == "coords_big":
        r = R(21)
        return np.column_stack([r.runif(400, -75, -35), r.runif(400, -33, 5)])
    if name == "env3":  # matrix(rnorm(150), 50, 3) + rep(c(0, 3, 6, 0, 3), each = 10)
        return R(12).rnorm(150).reshape(3, 50).T + np.repeat([0.0, 3.0, 6.0, 0.0, 3.0], 10)[:, None]
    if name == "dups":
        r = R(13)
        b = np.column_stack([r.runif(25, 0, 10), r.runif(25, 0, 10)])
        return b[np.r_[0:25, 0:10, 4:15, 2, 2, 2]]
    if name == "near":  # rows that differ from others only beyond 15 significant digits (distinct in R)
        r = R(14)
        b = np.column_stack([r.runif(30, -50, 50), r.runif(30, -50, 50)])
        return np.vstack([b, b[:10] * (1 + 2.0**-52), b[10:20] + 1e-13])
    if name == "ties":  # values next to 15-digit ties, with their neighbouring doubles
        d = np.floor(R(16).runif(20, 1e14, 1e15))
        t = (d + 0.5) / 1e13
        return np.column_stack([np.concatenate([t, t * (1 + 2.0**-52), t * (1 - 2.0**-52)]), np.tile([1.0, 2.0], 30)])
    if name == "oned":
        u = R(15).runif(40)
        return np.concatenate([u, u[:10], u[10:15] * (1 + 2.0**-52)])[:, None]
    if name == "qtran":  # many ties far from 0: the quick-transfer stage cycles
        return (1e12 + np.array([int(c) for c in QTRAN]) * 0.001)[:, None]
    if name == "tiny":  # (1e-170)^2 underflows: an empty cluster
        return np.array([[0.0, 0.0], [1e-170, 0.0], [5.0, 5.0], [6.0, 6.0]])
    raise KeyError(name)


def init_centers(x: np.ndarray, spec: str) -> np.ndarray:
    if spec == "first3":
        return x[:3]
    return np.vstack([x[0], x[1], [1000.0, 1000.0]])  # "far": a centre no point is near


def fromhex(values) -> np.ndarray:
    return np.array([float.fromhex(v) for v in values])


def same_bits(a, b) -> bool:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return a.shape == b.shape and bool(np.all((a == b) | (np.isnan(a) & np.isnan(b))))


CASES = [
    dict(data="coords", k=2, nstart=25, iter_max=500, algorithm="Hartigan-Wong", seed=1, 
         cluster="212112212112221222122212122112212112121211222211121221121212",
         centers=["-0x1.b4fe6b5015555p+5", "-0x1.9373e22ada12fp+2", "-0x1.8d442d96df07cp+5", "-0x1.d5fc3ddbaaaabp+3"],
         withinss=["0x1.5dd0ee514e0a7p+9", "0x1.e8585977fdeadp+9"], size=[27, 33], iter=1, ifault=0,
         totss=["0x1.81c448bee907p+11"], tot_withinss=["0x1.a314a3e4a5faap+10"], betweenss=["0x1.6073ed992c136p+10"],
         warnings=[], after=["0x1.c09c873p-2"]),
    dict(data="coords", k=3, nstart=25, iter_max=500, algorithm="Hartigan-Wong", seed=2, 
         cluster="121223123221333311111321211221121233212122131332332112212321",
         centers=["-0x1.a46c3c03b642dp+5", "-0x1.e47bd9c54de9cp+3", "-0x1.c01a06913a2e9p+5", "-0x1.90331c65ae8bap+2", "-0x1.66b5f3016p+5", "-0x1.68b85fd5p+3"],
         withinss=["0x1.6310ff555df1cp+8", "0x1.1e824259bf283p+8", "0x1.a81c306c1dab3p+8"], size=[23, 22, 15], iter=2, ifault=0,
         totss=["0x1.81c448bee907p+11"], tot_withinss=["0x1.0a6bdc86ceb14p+10"], betweenss=["0x1.f91cb4f7035ccp+10"],
         warnings=[], after=["0x1.58103e58p-1"]),
    dict(data="coords", k=4, nstart=25, iter_max=500, algorithm="Hartigan-Wong", seed=3, 
         cluster="313112312113224433333433133113313142131311323241424231134212",
         centers=["-0x1.c5d6473dabca2p+5", "-0x1.95dc03cbd7943p+2", "-0x1.62f05d7551746p+5", "-0x1.da7df43eae8bap+3", "-0x1.a7c20fc418ba3p+5", "-0x1.d862190f5745dp+3", "-0x1.7e56d12d94p+5", "-0x1.7427a72p+2"],
         withinss=["0x1.5bd017309bd25p+7", "0x1.fb6ff9c600b2ap+6", "0x1.465e8bfec54ffp+8", "0x1.080ac2c26d914p+7"], size=[19, 11, 22, 8], iter=3, ifault=0,
         totss=["0x1.81c448bee907p+11"], tot_withinss=["0x1.7b93fbb4e5173p+9"], betweenss=["0x1.22df49d1afc13p+11"],
         warnings=[], after=["0x1.9b650284p-2"]),
    dict(data="coords", k=5, nstart=25, iter_max=500, algorithm="Hartigan-Wong", seed=4, 
         cluster="412113413112335224424224124114214152141411224351235221125312",
         centers=["-0x1.c5d6473dabca2p+5", "-0x1.95dc03cbd7943p+2", "-0x1.8c398b6365a5ap+5", "-0x1.99ed885cda5a6p+3", "-0x1.546512dd5b6dbp+5", "-0x1.d5d2269b8p+3", "-0x1.b697952cdp+5", "-0x1.048e67029aaabp+4", "-0x1.7cce3a834p+5", "-0x1.d7ac64aep+1"],
         withinss=["0x1.5bd017309bd25p+7", "0x1.51236553be7ffp+7", "0x1.4b561290a1344p+5", "0x1.f77f5a8356577p+6", "0x1.053837b4a640fp+6"], size=[19, 17, 7, 12, 5], iter=3, ifault=0,
         totss=["0x1.81c448bee907p+11"], tot_withinss=["0x1.1f893291203aep+9"], betweenss=["0x1.39e1fc1aa0f84p+11"],
         warnings=[], after=["0x1.c5b03c44p-2"]),
    dict(data="coords", k=6, nstart=25, iter_max=500, algorithm="Hartigan-Wong", seed=5, 
         cluster="351556326552664211313221521551251541535355221645264125514651",
         centers=["-0x1.9680566d47627p+5", "-0x1.f8b458ec76276p+3", "-0x1.90e571ec5a2e9p+5", "-0x1.59274bb28p+3", "-0x1.c85bee70dp+5", "-0x1.096fe4b6aaaabp+4", "-0x1.7cce3a834p+5", "-0x1.d7ac64aep+1", "-0x1.c6f60970c5555p+5", "-0x1.88be7c231c71cp+2", "-0x1.546512dd5b6dbp+5", "-0x1.d5d2269b8p+3"],
         withinss=["0x1.5804631e5744cp+6", "0x1.14ee64a6dc5dep+6", "0x1.49482c1f3296bp+5", "0x1.053837b4a640fp+6", "0x1.31959c8c9673bp+7", "0x1.4b561290a1344p+5"], size=[13, 11, 6, 5, 18, 7], iter=4, ifault=0,
         totss=["0x1.81c448bee907p+11"], tot_withinss=["0x1.c7e955fabc2c2p+8"], betweenss=["0x1.48c71dff91818p+11"],
         warnings=[], after=["0x1.dacbc588p-1"]),
    dict(data="coords", k=7, nstart=25, iter_max=500, algorithm="Hartigan-Wong", seed=6, 
         cluster="742445735443551322727332432442362412674764332514356236426542",
         centers=["-0x1.6b7f093c6p+5", "-0x1.dd246c68p+1", "-0x1.9680566d47627p+5", "-0x1.f8b458ec76276p+3", "-0x1.90e571ec5a2e9p+5", "-0x1.59274bb28p+3", "-0x1.cd1cc62194925p+5", "-0x1.aa14e8958p+2", "-0x1.546512dd5b6dbp+5", "-0x1.d5d2269b8p+3", "-0x1.a88b4f7ddp+5", "-0x1.0549649055555p+2", "-0x1.c85bee70dp+5", "-0x1.096fe4b6aaaabp+4"],
         withinss=["0x1.2656c2737d7a8p+4", "0x1.5804631e5744cp+6", "0x1.14ee64a6dc5dep+6", "0x1.801a1bcb66a89p+6", "0x1.4b561290a1344p+5", "0x1.da88174fe8312p+4", "0x1.49482c1f3296bp+5"], size=[3, 13, 11, 14, 7, 6, 6], iter=3, ifault=0,
         totss=["0x1.81c448bee907p+11"], tot_withinss=["0x1.7de4ee565766ep+8"], betweenss=["0x1.5207aaf41e1a2p+11"],
         warnings=[], after=["0x1.6325e51p-3"]),
    dict(data="coords", k=8, nstart=25, iter_max=500, algorithm="Hartigan-Wong", seed=7, 
         cluster="742443763441335126627162412442182652876784112354138218428362",
         centers=["-0x1.8af0ab7eep+5", "-0x1.5f36bae5d5555p+3", "-0x1.94c8d9ec9aaabp+5", "-0x1.ff27a74615555p+3", "-0x1.546512dd5b6dbp+5", "-0x1.d5d2269b8p+3", "-0x1.d0febb77a8ba3p+5", "-0x1.79e7d9a65d174p+2", "-0x1.6b7f093c6p+5", "-0x1.dd246c68p+1", "-0x1.b7b34b3029249p+5", "-0x1.532b69ac49249p+3", "-0x1.c8ad1209p+5", "-0x1.15d23cb7cp+4", "-0x1.a88b4f7ddp+5", "-0x1.0549649055555p+2"],
         withinss=["0x1.3749782932c6fp+5", "0x1.21ed54e3effddp+6", "0x1.4b561290a1344p+5", "0x1.9504e58b46fa7p+5", "0x1.2656c2737d7a8p+4", "0x1.1b6b9803c98f8p+5", "0x1.723f296af396dp+4", "0x1.da88174fe8312p+4"], size=[9, 12, 7, 11, 3, 7, 5, 6], iter=4, ifault=0,
         totss=["0x1.81c448bee907p+11"], tot_withinss=["0x1.360f3674fe244p+8"], betweenss=["0x1.5b0261f049428p+11"],
         warnings=[], after=["0x1.780332aep-1"]),
    dict(data="coords", k=5, nstart=1, iter_max=500, algorithm="Hartigan-Wong", seed=8, 
         cluster="434332432334225544444544344334414352143413424253525241341232",
         centers=["-0x1.ad8b5e4bap+5", "-0x1.dcb78be6p+1", "-0x1.62f05d7551746p+5", "-0x1.da7df43eae8bap+3", "-0x1.cb474c1955555p+5", "-0x1.b878d174aaaabp+2", "-0x1.a7c20fc418ba3p+5", "-0x1.d862190f5745dp+3", "-0x1.79d1132609249p+5", "-0x1.9cd05681b6db7p+2"],
         withinss=["0x1.c2d15cbc79bbcp+3", "0x1.fb6ff9c600b2ap+6", "0x1.d6b609cd47882p+6", "0x1.465e8bfec54ffp+8", "0x1.6e181cc5d2581p+6"], size=[5, 11, 15, 22, 7], iter=3, ifault=0,
         totss=["0x1.81c448bee907p+11"], tot_withinss=["0x1.52424f7d77e14p+9"], betweenss=["0x1.2d33b4df8b0ebp+11"],
         warnings=[], after=["0x1.7017440cp-1"]),
    dict(data="coords", k=3, nstart=1, iter_max=10, algorithm="Hartigan-Wong", seed=9, 
         cluster="212113213112333322222312122112212133121211232331331221121312",
         centers=["-0x1.c01a06913a2e9p+5", "-0x1.90331c65ae8bap+2", "-0x1.a46c3c03b642dp+5", "-0x1.e47bd9c54de9cp+3", "-0x1.66b5f3016p+5", "-0x1.68b85fd5p+3"],
         withinss=["0x1.1e824259bf283p+8", "0x1.6310ff555df1cp+8", "0x1.a81c306c1dab3p+8"], size=[22, 23, 15], iter=3, ifault=0,
         totss=["0x1.81c448bee907p+11"], tot_withinss=["0x1.0a6bdc86ceb14p+10"], betweenss=["0x1.f91cb4f7035ccp+10"],
         warnings=[], after=["0x1.b9d282ep-3"]),
    dict(data="coords", k=1, nstart=1, iter_max=10, algorithm="Hartigan-Wong", seed=10, 
         cluster="111111111111111111111111111111111111111111111111111111111111",
         centers=["-0x1.9f24c95d6aaabp+5", "-0x1.5d44ce8275555p+3"],
         withinss=["0x1.81c448bee906fp+11"], size=[60], iter=1, ifault=None,
         totss=["0x1.81c448bee907p+11"], tot_withinss=["0x1.81c448bee906fp+11"], betweenss=["0x1p-41"],
         warnings=[], after=["0x1.3a2185f4p-2"]),
    dict(data="coords_big", k=5, nstart=25, iter_max=500, algorithm="Hartigan-Wong", seed=2, 
         cluster="4532442534135125121323132314113114425452141521532444345423522413154444134114454334532112411513351455534325251332112534445242252344315143242512213453413533322442351425111254545155451435332431233522152253521555141325423541234222242115511222122524333535322353223241322134114454125325445224442355445523111222511335123421335415132155243335342355422552255423251414211223112222351513124535312254343525341345",
         centers=["-0x1.9a02f135e6666p+5", "-0x1.9d8f9b914369dp+3", "-0x1.04b7259be10e1p+6", "-0x1.b6dd660373573p+1", "-0x1.6679e929fa2e9p+5", "-0x1.a1f92a7da710ap+4", "-0x1.4d68b9be77943p+5", "-0x1.e47d044bp+1", "-0x1.0e89204f9c71cp+6", "-0x1.7a641f37f5ba8p+4"],
         withinss=["0x1.555caacf7de14p+11", "0x1.3a7a97ef7fd2bp+12", "0x1.064c9171e354cp+12", "0x1.f175fa3cbf9cap+11", "0x1.0361d3acf0de3p+12"], size=[75, 91, 77, 76, 81], iter=5, ifault=0,
         totss=["0x1.8e4db5e488b95p+16"], tot_withinss=["0x1.39e493e51cb12p+14"], betweenss=["0x1.3fd490eb418dp+16"],
         warnings=[], after=["0x1.ae45218ap-1"]),
    dict(data="coords_big", k=8, nstart=25, iter_max=500, algorithm="Hartigan-Wong", seed=42, 
         cluster="5861554835364318212212331625336235314547337842867555258512811346785557225725545623867774532836643588833372743361731465534731181635634252751874472543532822611571643572277785254348344568227367166871781782877842273614516837165117737378832441717415222824611283176177677365735385274348358715551342538476373141837664346713268372767788756668651628511847128372743733137712741147287822775834621725256478353658",
         centers=["-0x1.13d6d26b32621p+6", "-0x1.de23ce48415cap+0", "-0x1.af6856856343fp+5", "-0x1.7f09e97dfb6dbp+4", "-0x1.5bb21668c22b6p+5", "-0x1.bd90b635b2fbbp+3", "-0x1.0b4d7149b8p+6", "-0x1.e87fd7bd68p+3", "-0x1.4ca090ee22762p+5", "-0x1.43fd36a2d89d9p+0", "-0x1.4b08d98092bb5p+5", "-0x1.b711defc512bbp+4", "-0x1.c078f370b64d9p+5", "-0x1.5b9aefea41f08p+2", "-0x1.158eb9993a6f5p+6", "-0x1.b41091a60b216p+4"],
         withinss=["0x1.7dca8d3e2b0dcp+10", "0x1.ab24390bfe808p+10", "0x1.de34f0e231c61p+10", "0x1.3b9509139a339p+10", "0x1.7c604197c2ff4p+10", "0x1.e0fd5288a25c4p+9", "0x1.2b696eb44f5f9p+11", "0x1.caabcb0c0c2fap+9"], size=[47, 49, 59, 40, 52, 41, 66, 46], iter=3, ifault=0,
         totss=["0x1.8e4db5e488b95p+16"], tot_withinss=["0x1.7d780dc155d18p+13"], betweenss=["0x1.5e9eb42c5dff2p+16"],
         warnings=[], after=["0x1.6e014da6p-1"]),
    dict(data="coords_big", k=6, nstart=25, iter_max=500, algorithm="Lloyd", seed=3, 
         cluster="1652112651456426434525453541445441126363414624653113516125622145361113451343161551653342144645564166651536324552442651116312262511546415312642245165145655522133544136433361612466164156553154355632462365633666434526125614251223313446644232423621555656522564335244533451441161436526116321112566116635444222643556425324556146453466315556512566132663266135324341244335443223563645431652542361515236514516",
         centers=["-0x1.441a80afdca82p+5", "-0x1.1338d6f8f0b76p+2", "-0x1.17ad8221412f7p+6", "-0x1.4c89718b1c71cp+2", "-0x1.ce2f26a0f9612p+5", "-0x1.3223aaabe5847p+1", "-0x1.97b8bf6edb6dbp+5", "-0x1.b9dead09b3333p+3", "-0x1.666e4237c0d79p+5", "-0x1.a3a292b288p+4", "-0x1.0dbe5a1935555p+6", "-0x1.8623acf2d7e4bp+4"],
         withinss=["0x1.6eeca1afde43p+11", "0x1.1236af27b621cp+11", "0x1.004adc4a463p+11", "0x1.281964c99c83dp+11", "0x1.02578e0be451ap+12", "0x1.aca42d1fa169ep+11"], size=[67, 54, 58, 70, 76, 75], iter=10, ifault=None,
         totss=["0x1.8e4db5e488b95p+16"], tot_withinss=["0x1.0b5b5b645c24bp+14"], betweenss=["0x1.4b76df0b71b02p+16"],
         warnings=[], after=["0x1.806ad5c8p-3"]),
    dict(data="coords_big", k=6, nstart=25, iter_max=500, algorithm="MacQueen", seed=3, 
         cluster="1652112651456426434525453541445441126143414624653113516125622145361113451343161551653342144645564166651536324552442651116312262511546415312642245165145655522133544136433361612466164156553154355632462365633666434526125614251223313446644232423631555656522564335214533451441161436526116321112566116635444222643556425324556146453466315556512566132663266135324341244335443223563645431652542361515236514516",
         centers=["-0x1.4614aa3c5bd3ap+5", "-0x1.104265af99f8ap+2", "-0x1.182380458304ap+6", "-0x1.4d8a8d0967d93p+2", "-0x1.d03ac7a09dcaep+5", "-0x1.3bf4862f00001p+1", "-0x1.9926617552492p+5", "-0x1.be8601728af87p+3", "-0x1.666e4237c0d79p+5", "-0x1.a3a292b288001p+4", "-0x1.0e0f3330960dep+6", "-0x1.8805fd9faa60cp+4"],
         withinss=["0x1.83229a342f53p+11", "0x1.0d6023e4cb70dp+11", "0x1.f4cb908b4d162p+10", "0x1.2d1a30a4fe932p+11", "0x1.02578e0be4519p+12", "0x1.9eb28a5a9e904p+11"], size=[69, 53, 58, 70, 76, 74], iter=5, ifault=None,
         totss=["0x1.8e4db5e488b95p+16"], tot_withinss=["0x1.0b6c8baec0e2bp+14"], betweenss=["0x1.4b7292f8d880ap+16"],
         warnings=[], after=["0x1.806ad5c8p-3"]),
    dict(data="coords_big", k=4, nstart=25, iter_max=1, algorithm="Hartigan-Wong", seed=5, 
         cluster="1432112431134124323323132331133331124142131423432111314123422123141111331131141331432122113413341144431324241332132431114212242311334313212412213143133433322112341124411241414144144134331131233422142243422444311324123412231222212114414222122421333434322343223211322131111141324324114221112344114423311222411334323121334114132144213334312344122442244123241111231223242222341433121434332241313424313314",
         centers=["-0x1.67f9d6edbaaabp+5", "-0x1.80813838fdddep+2", "-0x1.03a030ea086bdp+6", "-0x1.f49953cf66666p+1", "-0x1.6f850ba4a999ap+5", "-0x1.831b491cb9eb8p+4", "-0x1.0cc6370110f0fp+6", "-0x1.757ce8372cccdp+4"],
         withinss=["0x1.faf4d5671ab5fp+12", "0x1.5851716486267p+12", "0x1.92155eb2ccd11p+12", "0x1.230f74b89fa06p+12"], size=[120, 95, 100, 85], iter=2, ifault=2,
         totss=["0x1.8e4db5e488b95p+16"], tot_withinss=["0x1.821ac68dc3537p+14"], betweenss=["0x1.2dc7044117e47p+16"],
         warnings=["did not converge in 1 iteration"] * 25, after=["0x1.a03aadp-4"]),
    dict(data="coords_big", k=4, nstart=25, iter_max=2, algorithm="Lloyd", seed=5, 
         cluster="1432114431134124324323132331133431124142131443432111314123422143241112331132141331432122113413341144431324241332132431114212242311334313212414413143133433322112341124411241414344344134332331233422142244422444311324123414231222232114414222122421333444322343223211322131111141424344114221112344114423311222412334323121334114132144213334312344122442244123241211231223142242341433121434332241413424313314",
         centers=["-0x1.6721e059d8376p+5", "-0x1.84c08c798a60ep+2", "-0x1.0065c6fd38591p+6", "-0x1.6726ae47f4deap+1", "-0x1.68b9583485e0ap+5", "-0x1.7e51cbe8bd634p+4", "-0x1.098b63a1bdbccp+6", "-0x1.671b0f408c9b2p+4"],
         withinss=["0x1.b9f6add6d82e8p+12", "0x1.56dafa2da5691p+12", "0x1.82206d181e1dfp+12", "0x1.a45fad3eaac51p+12"], size=[111, 92, 98, 99], iter=3, ifault=2,
         totss=["0x1.8e4db5e488b95p+16"], tot_withinss=["0x1.8dd47096d19eap+14"], betweenss=["0x1.2ad899bed451ap+16"],
         warnings=["did not converge in 2 iterations"] * 25, after=["0x1.a03aadp-4"]),
    dict(data="coords_big", k=4, nstart=3, iter_max=1, algorithm="MacQueen", seed=5, 
         cluster="2124234122221241242242224223222223241333222142123233231242144232312323223323212222123334222122212311122232412224324122321424414223221222334134432312322122244234212331233413121211213321223223422134314312133111232241342123423444323231132444343143222111244212432433233223323213231241231443334211321132233444133221242342221231323311432221234211344114411242313333423342314443213122343121224413232141232231",
         centers=["-0x1.0e552be2fa12fp+6", "-0x1.7c2d1e1bbc71ap+4", "-0x1.67090f898dbefp+5", "-0x1.38d0898ad2786p+4", "-0x1.9075902bd3a4fp+5", "-0x1.fff1f9bff865ap+1", "-0x1.0dce00cc62519p+6", "-0x1.d7e02c4437a6ep+1"],
         withinss=["0x1.0c6978afee2dfp+12", "0x1.e591ef7f64affp+13", "0x1.fc54391717deep+12", "0x1.7f81528b376b7p+11"], size=[81, 149, 101, 69], iter=2, ifault=2,
         totss=["0x1.8e4db5e488b95p+16"], tot_withinss=["0x1.e4e88e82dac8ap+14"], betweenss=["0x1.15139243d2072p+16"],
         warnings=["did not converge in 1 iteration"] * 3, after=["0x1.0ce5373cp-2"]),
    dict(data="env3", k=3, nstart=25, iter_max=500, algorithm="Hartigan-Wong", seed=1, 
         cluster="22222222223333333333111111111122222222223333333333",
         centers=["0x1.8d5c5909c2945p+2", "0x1.6d2e7e73134c7p+2", "0x1.7d42a7cfc75eep+2", "-0x1.ec2aab9a61536p-3", "0x1.08c0d22895e07p-4", "0x1.7964ac66024c9p-2", "0x1.63a985ee75562p+1", "0x1.a4559a5f9255dp+1", "0x1.603a18ea6ca2cp+1"],
         withinss=["0x1.69704b2cd8acp+4", "0x1.8382e888acc25p+5", "0x1.7f5115789b71dp+5"], size=[10, 20, 20], iter=2, ifault=0,
         totss=["0x1.a935f3255e89ap+9"], tot_withinss=["0x1.dbc611cbda451p+6"], betweenss=["0x1.6dbd30ebe341p+9"],
         warnings=[], after=["0x1.e2ca58ep-5"]),
    dict(data="env3", k=6, nstart=25, iter_max=500, algorithm="Hartigan-Wong", seed=2, 
         cluster="35335355532226226266111441114155535353556266222222",
         centers=["0x1.9de48ad5d7dcp+2", "0x1.630159b15e341p+2", "0x1.a5cf8da80e2e5p+2", "0x1.4f75f32ccf641p+1", "0x1.80108a0b8f9a5p+1", "0x1.1494439229beap+1", "-0x1.4f26d2fbbddf4p-1", "0x1.aa2793b4a1c42p-1", "0x1.11106b80508a1p+0", "0x1.66c93a2d90ed1p+2", "0x1.84ed298c64855p+2", "0x1.1ea48f81ccd05p+2", "0x1.25d6fd22b5cbdp-5", "-0x1.c9e46d5fee92bp-2", "-0x1.8cbaa805f66d3p-4", "0x1.892ddf9f3bce9p+1", "0x1.e7b14afb9768ap+1", "0x1.ecb73769c44a5p+1"],
         withinss=["0x1.3dbf7d2e5c443p+3", "0x1.597f95d050a9fp+4", "0x1.2d50c8530e056p+3", "0x1.2573e8836f5d1p+0", "0x1.660d7a5cf0934p+4", "0x1.2e16e6699a23cp+3"], size=[7, 13, 8, 3, 12, 7], iter=3, ifault=0,
         totss=["0x1.a935f3255e89ap+9"], tot_withinss=["0x1.279df92abe9a7p+6"], betweenss=["0x1.8442340006b65p+9"],
         warnings=[], after=["0x1.187081e8p-2"]),
    dict(data="env3", k=4, nstart=1, iter_max=500, algorithm="Hartigan-Wong", seed=3, 
         cluster="13113133314444444444222222222233313131334444444444",
         centers=["-0x1.4f26d2fbbddf4p-1", "0x1.aa2793b4a1c42p-1", "0x1.11106b80508a1p+0", "0x1.8d5c5909c2945p+2", "0x1.6d2e7e73134c7p+2", "0x1.7d42a7cfc75eep+2", "0x1.25d6fd22b5cbdp-5", "-0x1.c9e46d5fee92bp-2", "-0x1.8cbaa805f66d3p-4", "0x1.63a985ee75562p+1", "0x1.a4559a5f9255dp+1", "0x1.603a18ea6ca2cp+1"],
         withinss=["0x1.2d50c8530e056p+3", "0x1.69704b2cd8acp+4", "0x1.660d7a5cf0934p+4", "0x1.7f5115789b71dp+5"], size=[8, 10, 12, 20], iter=3, ifault=0,
         totss=["0x1.a935f3255e89ap+9"], tot_withinss=["0x1.9932152921c96p+6"], betweenss=["0x1.760fb0803a507p+9"],
         warnings=[], after=["0x1.3573233p-1"]),
    dict(data="env3", k=4, nstart=25, iter_max=500, algorithm="Forgy", seed=4, 
         cluster="11111111114442442422333333333311111111112422444444",
         centers=["-0x1.ec2aab9a61536p-3", "0x1.08c0d22895e07p-4", "0x1.7964ac66024c9p-2", "0x1.892ddf9f3bce9p+1", "0x1.e7b14afb9768ap+1", "0x1.ecb73769c44a5p+1", "0x1.8d5c5909c2945p+2", "0x1.6d2e7e73134c7p+2", "0x1.7d42a7cfc75eep+2", "0x1.4f75f32ccf641p+1", "0x1.80108a0b8f9a5p+1", "0x1.1494439229beap+1"],
         withinss=["0x1.8382e888acc22p+5", "0x1.2e16e6699a23bp+3", "0x1.69704b2cd8ac2p+4", "0x1.597f95d050aap+4"], size=[20, 7, 10, 13], iter=8, ifault=None,
         totss=["0x1.a935f3255e89ap+9"], tot_withinss=["0x1.98404950d3fb1p+6"], betweenss=["0x1.762de9fb440a4p+9"],
         warnings=[], after=["0x1.6c67353cp-2"]),
    dict(data="env3", k=5, nstart=25, iter_max=500, algorithm="MacQueen", seed=5, 
         cluster="41441411143332332322555555555511141414112322333333",
         centers=["0x1.25d6fd22b5cbbp-5", "-0x1.c9e46d5fee928p-2", "-0x1.8cbaa805f66d2p-4", "0x1.892ddf9f3bce9p+1", "0x1.e7b14afb9768ap+1", "0x1.ecb73769c44a4p+1", "0x1.4f75f32ccf641p+1", "0x1.80108a0b8f9a7p+1", "0x1.1494439229beap+1", "-0x1.4f26d2fbbddf3p-1", "0x1.aa2793b4a1c41p-1", "0x1.11106b80508a1p+0", "0x1.8d5c5909c2945p+2", "0x1.6d2e7e73134c7p+2", "0x1.7d42a7cfc75eep+2"],
         withinss=["0x1.660d7a5cf0935p+4", "0x1.2e16e6699a23bp+3", "0x1.597f95d050a9fp+4", "0x1.2d50c8530e055p+3", "0x1.69704b2cd8ac2p+4"], size=[12, 7, 13, 8, 10], iter=4, ifault=None,
         totss=["0x1.a935f3255e89ap+9"], tot_withinss=["0x1.55ac4cae1b7f8p+6"], betweenss=["0x1.7e80698f9b19bp+9"],
         warnings=[], after=["0x1.e8f8f288p-3"]),
    dict(data="env3", k=2, nstart=25, iter_max=500, algorithm="Hartigan-Wong", seed=6, 
         cluster="11111111112222222222222222222211111111112122222221",
         centers=["-0x1.52c8261e6faa3p-6", "0x1.ed740945c6ae9p-3", "0x1.008e85cd85dd4p-1", "0x1.02fd9da202dadp+2", "0x1.0f615669663c3p+2", "0x1.fb386358067c9p+1"],
         withinss=["0x1.19b30713ac4dbp+6", "0x1.cefd5c334b80fp+7"], size=[22, 28], iter=1, ifault=0,
         totss=["0x1.a935f3255e89ap+9"], tot_withinss=["0x1.2deb6fde90d3ep+8"], betweenss=["0x1.12403b36161fbp+9"],
         warnings=[], after=["0x1.60a9c7cp-1"]),
    dict(data="dups", k=4, nstart=25, iter_max=500, algorithm="Hartigan-Wong", seed=1, 
         cluster="4321412441343432244413241432141244141244134343222",
         centers=["0x1.200ae8d6p-1", "0x1.498d8b898p+2", "0x1.2708b92b22e8cp+2", "0x1.1099b425f45d1p+2", "0x1.7affda6271c72p+2", "0x1.8ccc14d78e38ep-1", "0x1.f1af1eeaf9436p+2", "0x1.cd3a8332650d8p+2"],
         withinss=["0x1.e4ef7efb87abp+4", "0x1.5cc2134b8aa71p+4", "0x1.6121918b3271cp+5", "0x1.34e9d1f7a7aedp+6"], size=[10, 11, 9, 19], iter=2, ifault=0,
         totss=["0x1.87270d31d8e3p+9"], tot_withinss=["0x1.5af37fa782be2p+7"], betweenss=["0x1.306a2d47f8338p+9"],
         warnings=[], after=["0x1.8277e40ep-1"]),
    dict(data="dups", k=2, nstart=25, iter_max=500, algorithm="Hartigan-Wong", seed=2, 
         cluster="2111211221121211122211121211121122121122112121111",
         centers=["0x1.d3c2f38eaaaabp+1", "0x1.c15e62b255555p+1", "0x1.f1af1eeaf9436p+2", "0x1.cd3a8332650d8p+2"],
         withinss=["0x1.5c870e2a1a9aap+8", "0x1.34e9d1f7a7aedp+6"], size=[30, 19], iter=1, ifault=0,
         totss=["0x1.87270d31d8e3p+9"], tot_withinss=["0x1.a9c182a804865p+8"], betweenss=["0x1.648c97bbad3fbp+8"],
         warnings=[], after=["0x1.fa516222p-1"]),
    dict(data="dups", k=6, nstart=1, iter_max=500, algorithm="Hartigan-Wong", seed=3, 
         cluster="2653235113626165512246514265323511323511362616555",
         centers=["0x1.ca96931bcp+2", "0x1.0cc5acc32p+3", "0x1.0e8fdd218p+3", "0x1.786eea6371c72p+2", "0x1.b19c0628p-2", "0x1.1bf8a7ad9p+2", "0x1.1e7f3fefp+0", "0x1.ffe11af94p+2", "0x1.2708b92b22e8cp+2", "0x1.1099b425f45d1p+2", "0x1.7affda6271c72p+2", "0x1.8ccc14d78e38ep-1"],
         withinss=["0x1.7efcdb0377a72p+4", "0x1.ecaa607798dc3p+3", "0x1.91b42384ad43dp+2", "0x1.7bdb22b3f47ecp+1", "0x1.5cc2134b8aa71p+4", "0x1.6121918b3271cp+5"], size=[10, 9, 8, 2, 11, 9], iter=3, ifault=0,
         totss=["0x1.87270d31d8e3p+9"], tot_withinss=["0x1.ca0febb637602p+6"], betweenss=["0x1.4de50fbb11f7p+9"],
         warnings=[], after=["0x1.c87ee85p-4"]),
    dict(data="dups", k=6, nstart=1, iter_max=500, algorithm="Hartigan-Wong", seed=4, 
         cluster="3166365446535252143325526316636544636544653525666",
         centers=["0x1.6a888744aaaabp+1", "0x1.f4bc7592aaaabp-1", "0x1.14c3bd0ap+2", "0x1.e2df4b4b8p+2", "0x1.0e8fdd218p+3", "0x1.786eea6371c72p+2", "0x1.f74bba6p+2", "0x1.191ba50c5b6dbp+3", "0x1.a521bfe845d17p+2", "0x1.fb11156a5d174p+0", "0x1.b20bc4c86db6ep+0", "0x1.2ad9701c89249p+2"],
         withinss=["0x1.b8f71ea3b8c23p+1", "0x1.2b85e7b8bd754p+4", "0x1.ecaa607798dc3p+3", "0x1.1e11873327cffp+3", "0x1.33273a672872p+5", "0x1.891c69dd83dc5p+5"], size=[3, 5, 9, 7, 11, 14], iter=2, ifault=0,
         totss=["0x1.87270d31d8e3p+9"], tot_withinss=["0x1.0c1140fd7dbp+7"], betweenss=["0x1.4422bcf27977p+9"],
         warnings=[], after=["0x1.0335f04cp-1"]),
    dict(data="dups", k=6, nstart=1, iter_max=500, algorithm="Hartigan-Wong", seed=5, 
         cluster="2155253225324636162263365215525322525322532463555",
         centers=["0x1.6a888744aaaabp+1", "0x1.f4bc7592aaaabp-1", "0x1.0b27b5efcp+3", "0x1.bef9d20aaaaabp+2", "0x1.840f34ff8e38ep+2", "0x1.2717b09fe38e4p+1", "0x1.1cfa987fcp+3", "0x1.0a118054p-1", "0x1.b20bc4c86db6ep+0", "0x1.2ad9701c89249p+2", "0x1.25b9d6036aaabp+2", "0x1.f985bd00eaaabp+2"],
         withinss=["0x1.b8f71ea3b8c23p+1", "0x1.74a627991b784p+5", "0x1.3f526581fb553p+4", "0x0p+0", "0x1.891c69dd83dc5p+5", "0x1.895b678d966a8p+4"], size=[3, 15, 9, 2, 14, 6], iter=2, ifault=0,
         totss=["0x1.87270d31d8e3p+9"], tot_withinss=["0x1.1f6a3a7a28f02p+7"], betweenss=["0x1.3f4c7e934ea7p+9"],
         warnings=[], after=["0x1.e9ba6268p-1"]),
    dict(data="dups", k=6, nstart=1, iter_max=500, algorithm="Lloyd", seed=6, 
         cluster="5132523552656563155546652513252355252355265656333",
         centers=["0x1.6a888744aaaabp+1", "0x1.f4bc7592aaaabp-1", "0x1.e424b7938e38ep-2", "0x1.2cd1a5d8d5555p+2", "0x1.1ef959ea9c71cp+2", "0x1.256bf1570e38ep+2", "0x1.5ee3ef22p+0", "0x1.26144f5fcp+3", "0x1.f1af1eeaf9436p+2", "0x1.cd3a8332650d8p+2", "0x1.b939b5c54p+2", "0x1.34371676cp+0"],
         withinss=["0x1.b8f71ea3b8c24p+1", "0x1.6e1fd8a86c75p+3", "0x1.44a6ad65dd899p+3", "0x0p+0", "0x1.34e9d1f7a7aebp+6", "0x1.1cdd7ecb9c3dfp+4"], size=[3, 9, 9, 1, 19, 8], iter=4, ifault=None,
         totss=["0x1.87270d31d8e3p+9"], tot_withinss=["0x1.e041bb6175c41p+6"], betweenss=["0x1.4b1ed5c5aa2a8p+9"],
         warnings=[], after=["0x1.66ab598ap-1"]),
    dict(data="near", k=3, nstart=25, iter_max=500, algorithm="Hartigan-Wong", seed=1, 
         cluster="21323132221223312132113131221221323132221223312132",
         centers=["0x1.cfbc57242aab7p+3", "-0x1.1d82a91df5552p+5", "-0x1.457f7acc55549p+3", "0x1.bb80a36fb6dc3p+3", "0x1.2b19b0686b6dfp+5", "0x1.6e158a01f6dbep+4"],
         withinss=["0x1.d1f559af3448cp+11", "0x1.775b3f324d17p+13", "0x1.9ae72fcd66ad9p+12"], size=[15, 21, 14], iter=2, ifault=0,
         totss=["0x1.193ff8311abbep+16"], tot_withinss=["0x1.5ca616c266cp+14"], betweenss=["0x1.842ce5010217cp+15"],
         warnings=[], after=["0x1.e2ca58ep-5"]),
    dict(data="near", k=7, nstart=25, iter_max=500, algorithm="Hartigan-Wong", seed=2, 
         cluster="46737152326425562653113636236246737152326425562653",
         centers=["-0x1.4a697b91cp+2", "-0x1.13b4329358p+5", "-0x1.93da86423fff5p+3", "-0x1.acfe8fa9fffd3p+1", "0x1.d3dbc9ccaaab7p+2", "0x1.aed7f03b71c75p+4", "-0x1.d1bdb4e83fff9p+4", "0x1.35b80339c0004p+5", "0x1.2a9d024e28006p+5", "0x1.0285812f80015p+3", "0x1.5a388c775d17dp+4", "-0x1.21138e67a2e87p+5", "0x1.781a3140fp+5", "0x1.6e189c644p+5"],
         withinss=["0x1.eaf47341f308ep+7", "0x1.e37dffb8f0ec7p+10", "0x1.52db96c6c417ep+10", "0x1.37632724032c7p+8", "0x1.120d72a07ea0fp+10", "0x1.55baf8dacc81bp+10", "0x1.74e51a59a5813p+3"], size=[4, 10, 9, 4, 8, 11, 4], iter=3, ifault=0,
         totss=["0x1.193ff8311abbep+16"], tot_withinss=["0x1.8b10c9183ca79p+12"], betweenss=["0x1.008eeb9f96f16p+16"],
         warnings=[], after=["0x1.c9146f4p-1"]),
    dict(data="near", k=5, nstart=1, iter_max=500, algorithm="Hartigan-Wong", seed=3, 
         cluster="24131415354251145413543414534524131415354251145413",
         centers=["0x1.3add9512113b5p+5", "0x1.55dec4fde276ap+4", "-0x1.d1bdb4e83fff9p+4", "0x1.35b80339c0004p+5", "0x1.466788906000ep+2", "0x1.a8844b2250004p+4", "0x1.097785722925p+4", "-0x1.21ffa7cd29246p+5", "-0x1.9a81b3fef45c7p+3", "-0x1.64f15553e8b8fp+2"],
         withinss=["0x1.54ad33ac92d6bp+12", "0x1.37632724032c7p+8", "0x1.f03954990a095p+9", "0x1.55f0b1bbc2cb1p+11", "0x1.36747eed5f3c5p+11"], size=[13, 4, 8, 14, 11], iter=2, ifault=0,
         totss=["0x1.193ff8311abbep+16"], tot_withinss=["0x1.762e948342a73p+13"], betweenss=["0x1.d4f44b4164cdfp+15"],
         warnings=[], after=["0x1.2dabdd5cp-2"]),
    dict(data="ties", k=4, nstart=25, iter_max=500, algorithm="Hartigan-Wong", seed=1, 
         cluster="133323422431411321141333234224314113211413332342243141132114",
         centers=["0x1.0ff8a60f78p+6", "0x1.8p+0", "0x1.5b96027932001p+6", "0x1.4p+0", "0x1.276098d040003p+5", "0x1.aaaaaaaaaaaabp+0", "0x1.40f25d392fffbp+4", "0x1.8p+0"],
         withinss=["0x1.3de31c24509bfp+9", "0x1.b2153255cfb1ap+4", "0x1.a7e04ad1ebcbap+9", "0x1.112226f84faefp+7"], size=[18, 12, 18, 12], iter=2, ifault=0,
         totss=["0x1.22469a7625578p+15"], tot_withinss=["0x1.9bce4d237f687p+10"], betweenss=["0x1.1568280d095c4p+15"],
         warnings=[], after=["0x1.44593aaap-1"]),
    dict(data="ties", k=8, nstart=25, iter_max=500, algorithm="Hartigan-Wong", seed=2, 
         cluster="536381288217254385525363812882172543855253638128821725438552",
         centers=["0x1.3323ea2480004p+5", "0x1.8p+0", "0x1.40f25d392fffbp+4", "0x1.8p+0", "0x1.f69a26a92p+4", "0x1p+1", "0x1.ca3ed000bffffp+5", "0x1p+0", "0x1.1f79d5becbfffp+6", "0x1.8p+0", "0x1.9414869ad0005p+5", "0x1p+0", "0x1.f99a4ac27ffffp+5", "0x1p+1", "0x1.5b96027932001p+6", "0x1.4p+0"],
         withinss=["0x1.318c4a6eca424p+1", "0x1.112226f84faefp+7", "0x1.67971bde925ddp+1", "0x1p-91", "0x1.4699441873841p+5", "0x1.6p-91", "0x1p-91", "0x1.b2153255cfb1ap+4"], size=[6, 12, 9, 3, 12, 3, 3, 12], iter=1, ifault=0,
         totss=["0x1.22469a7625578p+15"], tot_withinss=["0x1.a36fabe25bf8bp+7"], betweenss=["0x1.20a32aca42fb8p+15"],
         warnings=[], after=["0x1.e747378p-2"]),
    dict(data="oned", k=3, nstart=25, iter_max=500, algorithm="Hartigan-Wong", seed=1, 
         cluster="3213211231233311312111223322133232332321321321123123331",
         centers=["0x1.bda54e7ef0f0fp-1", "0x1.9fc073fbc3c3cp-3", "0x1.2f0bfc44p-1"],
         withinss=["0x1.99d272091532cp-4", "0x1.53fec7d4989e6p-3", "0x1.2aa1118ade3ebp-3"], size=[17, 17, 21], iter=2, ifault=0,
         totss=["0x1.0f3304a2d7dd9p+2"], tot_withinss=["0x1.a5c4893200bb4p-2"], betweenss=["0x1.e9ad781f6fa3cp+1"],
         warnings=[], after=["0x1.35b9d0f4p-1"]),
    dict(data="oned", k=5, nstart=1, iter_max=500, algorithm="Hartigan-Wong", seed=2, 
         cluster="1531432512514122435222451155344545415452153143251251412",
         centers=["0x1.4d8940d0ec4ecp-1", "0x1.a29e81288ba2fp-1", "0x1.ef31c6f2aaaabp-1", "0x1.d65b90be2e8bap-2", "0x1.5923eb6p-3"],
         withinss=["0x1.585d1c2304bfp-6", "0x1.5ef3bd2900bedp-7", "0x1.47e083fbefc4cp-9", "0x1.533f3e72d52f9p-5", "0x1.2575c994294edp-4"], size=[13, 11, 6, 11, 14], iter=2, ifault=0,
         totss=["0x1.0f3304a2d7dd9p+2"], tot_withinss=["0x1.2da515cdaa563p-3"], betweenss=["0x1.05c5dbf46a8aep+2"],
         warnings=[], after=["0x1.e30f2658p-1"]),
    dict(data="oned", k=4, nstart=25, iter_max=500, algorithm="Lloyd", seed=3, 
         cluster="2342144324321244143444132233411313123132234214432432124",
         centers=["0x1.d65b90be2e8bap-2", "0x1.5184b66892492p-1", "0x1.5923eb6p-3", "0x1.c12b08752p-1"],
         withinss=["0x1.533f3e72d52f9p-5", "0x1.066069c8a7ef6p-5", "0x1.2575c994294edp-4", "0x1.65178a4fa2006p-4"], size=[11, 14, 14, 16], iter=6, ifault=None,
         totss=["0x1.0f3304a2d7dd9p+2"], tot_withinss=["0x1.dbae9400c4ef5p-3"], betweenss=["0x1.00559002d1b61p+2"],
         warnings=[], after=["0x1.1372ca92p-1"]),
    dict(data="qtran", k=3, nstart=1, iter_max=50, algorithm="Hartigan-Wong", seed=1, 
         cluster="1122122221121113111221111232333132313323233122231312333333312211121111111111121111111111133231111111",
         centers=["0x1.d1a94a2000005p+39", "0x1.d1a94a2p+39", "0x1.d1a94a200000bp+39"],
         withinss=["0x1.dfap-15", "0x0p+0", "0x1.e8p-18"], size=[53, 23, 24], iter=1, ifault=4,
         totss=["0x1.01p-14"], tot_withinss=["0x1.0e5p-14"], betweenss=["-0x1.aap-19"],
         warnings=["Quick-TRANSfer stage steps exceeded maximum (= 5000)"], after=["0x1.9d0b6a9p-3"]),
    dict(data="qtran", k=3, nstart=4, iter_max=50, algorithm="Hartigan-Wong", seed=2, 
         cluster="3322322223323331333223333212111312131121211322213132111111132233323333333333323333333333311213333333",
         centers=["0x1.d1a94a200000bp+39", "0x1.d1a94a2p+39", "0x1.d1a94a2000005p+39"],
         withinss=["0x1.e8p-18", "0x0p+0", "0x1.dfap-15"], size=[24, 23, 53], iter=1, ifault=4,
         totss=["0x1.01p-14"], tot_withinss=["0x1.0e5p-14"], betweenss=["-0x1.aap-19"],
         warnings=["Quick-TRANSfer stage steps exceeded maximum (= 5000)"] * 4, after=["0x1.7251cf18p-3"]),
    dict(data="tiny", k=3, nstart=1, iter_max=10, algorithm="Hartigan-Wong", seed=1, 
         cluster="1123",
         centers=["0x1.3529ba7d19eafp-566", "0x0p+0", "0x1.4p+2", "0x1.4p+2", "0x1.8p+2", "0x1.8p+2"],
         withinss=["0x0p+0", "0x0p+0", "0x0p+0"], size=[2, 1, 1], iter=1, ifault=0,
         totss=["0x1.ecp+5"], tot_withinss=["0x0p+0"], betweenss=["0x1.ecp+5"],
         warnings=[], after=["0x1.9d0b6a9p-3"]),
    dict(data="tiny", k=3, nstart=1, iter_max=10, algorithm="Hartigan-Wong", seed=2, error="empty cluster: try a better set of initial centers"),
    dict(data="tiny", k=3, nstart=1, iter_max=10, algorithm="Hartigan-Wong", seed=3, error="empty cluster: try a better set of initial centers"),
    dict(data="tiny", k=3, nstart=1, iter_max=10, algorithm="Hartigan-Wong", seed=4, 
         cluster="3321",
         centers=["0x1.8p+2", "0x1.8p+2", "0x1.4p+2", "0x1.4p+2", "0x1.3529ba7d19eafp-566", "0x0p+0"],
         withinss=["0x0p+0", "0x0p+0", "0x0p+0"], size=[1, 1, 2], iter=1, ifault=0,
         totss=["0x1.ecp+5"], tot_withinss=["0x0p+0"], betweenss=["0x1.ecp+5"],
         warnings=[], after=["0x1.1c082e54p-2"]),
    dict(data="dups", k=30, nstart=25, iter_max=10, algorithm="Hartigan-Wong", seed=1, error="more cluster centers than distinct data points."),
    dict(data="tiny", k=3, nstart=1, iter_max=10, algorithm="Hartigan-Wong", seed=1, init="first3", error="empty cluster: try a better set of initial centers"),
    dict(data="coords", k=3, nstart=1, iter_max=10, algorithm="Lloyd", seed=1, init="far", 
         cluster="121221121221112111211121211221121221212122111122112112212121",
         centers=["-0x1.8d35a0458f0f1p+5", "-0x1.d062310b6d2d3p+3", "-0x1.b6989ca4p+5", "-0x1.8d77d6639d89ep+2", "NaN", "NaN"],
         withinss=["0x1.f9909a2ec35c7p+9", "0x1.4caa97ad367fbp+9", "0x0p+0"], size=[34, 26, 0], iter=4, ifault=None,
         totss=["0x1.81c448bee907p+11"], tot_withinss=["0x1.a31d98edfcee1p+10"], betweenss=["0x1.606af88fd51ffp+10"],
         warnings=["empty cluster: try a better set of initial centers"] * 2, after=["0x1.0fe180c4p-2"]),
    dict(data="coords", k=3, nstart=1, iter_max=10, algorithm="MacQueen", seed=1, init="far", 
         cluster="121221121221112111211121211221121221212122111122112112212121",
         centers=["-0x1.8d35a0458f0f2p+5", "-0x1.d062310b6d2d3p+3", "-0x1.b6989ca400001p+5", "-0x1.8d77d6639d89ep+2", "NaN", "NaN"],
         withinss=["0x1.f9909a2ec35c8p+9", "0x1.4caa97ad367fbp+9", "0x0p+0"], size=[34, 26, 0], iter=3, ifault=None,
         totss=["0x1.81c448bee907p+11"], tot_withinss=["0x1.a31d98edfcee2p+10"], betweenss=["0x1.606af88fd51fep+10"],
         warnings=["empty cluster: try a better set of initial centers"] * 2, after=["0x1.0fe180c4p-2"]),
    dict(data="coords", k=3, nstart=1, iter_max=500, algorithm="Hartigan-Wong", seed=1, init="first3", 
         cluster="121223123221333311111321211221121233212122131332332112212321",
         centers=["-0x1.a46c3c03b642dp+5", "-0x1.e47bd9c54de9cp+3", "-0x1.c01a06913a2e9p+5", "-0x1.90331c65ae8bap+2", "-0x1.66b5f3016p+5", "-0x1.68b85fd5p+3"],
         withinss=["0x1.6310ff555df1cp+8", "0x1.1e824259bf283p+8", "0x1.a81c306c1dab3p+8"], size=[23, 22, 15], iter=2, ifault=0,
         totss=["0x1.81c448bee907p+11"], tot_withinss=["0x1.0a6bdc86ceb14p+10"], betweenss=["0x1.f91cb4f7035ccp+10"],
         warnings=[], after=["0x1.0fe180c4p-2"]),
]


def case_id(c: dict) -> str:
    return f"{c['data']}-k{c['k']}-n{c['nstart']}-i{c['iter_max']}-{c['algorithm']}-s{c['seed']}" + (
        f"-{c['init']}" if "init" in c else "")


@pytest.mark.parametrize("case", CASES, ids=[case_id(c) for c in CASES])
def test_kmeans_matches_r(case):
    x = make(case["data"])
    rng = _core.RRandom(case["seed"])
    kwargs = dict(iter_max=case["iter_max"], nstart=case["nstart"], algorithm=case["algorithm"], rng=rng)
    if "init" in case:
        kwargs["centers"] = init_centers(x, case["init"])
    else:
        kwargs["k"] = case["k"]
    if "error" in case:
        with pytest.raises((RuntimeError, ValueError), match=re.escape(case["error"])):
            _core.sdm_kmeans(x, **kwargs)
        return
    res = _core.sdm_kmeans(x, **kwargs)
    k = len(case["size"])
    assert "".join(map(str, res["cluster"])) == case["cluster"]
    assert same_bits(res["centers"], fromhex(case["centers"]).reshape(k, x.shape[1]))
    assert same_bits(res["withinss"], fromhex(case["withinss"]))
    assert res["size"].tolist() == case["size"]
    assert res["iter"] == case["iter"]
    assert res["ifault"] == case["ifault"]
    for key in ("totss", "tot_withinss", "betweenss"):
        assert same_bits(res[key], fromhex(case[key])[0]), key
    assert res["warnings"] == case["warnings"]
    # the same use of R's stream: the next draw is R's
    assert same_bits(rng.runif(1), fromhex(case["after"]))


def test_seed_argument_is_set_seed():
    x = make("coords")
    a = _core.sdm_kmeans(x, 4, iter_max=500, nstart=25, seed=3)
    b = _core.sdm_kmeans(x, 4, iter_max=500, nstart=25, rng=_core.RRandom(3))
    assert np.array_equal(a["cluster"], b["cluster"]) and same_bits(a["centers"], b["centers"])


def test_errors():
    x = make("coords")
    with pytest.raises(ValueError, match="cannot take a sample larger"):
        _core.sdm_kmeans(x, 61, seed=1)
    with pytest.raises(ValueError, match="between 1 and nrow"):
        _core.sdm_kmeans(x[:4], 4, seed=1)  # Hartigan-Wong needs k < nrow(x)
    with pytest.raises(ValueError, match="iter.max"):
        _core.sdm_kmeans(x, 3, iter_max=0, seed=1)
    with pytest.raises(ValueError, match="algorithm"):
        _core.sdm_kmeans(x, 3, seed=1, algorithm="Elkan")
    with pytest.raises(ValueError, match="not distinct"):
        _core.sdm_kmeans(x, centers=np.vstack([x[0], x[0]]))
    bad = x.copy()
    bad[3, 1] = np.nan
    with pytest.raises(ValueError, match="NA/NaN/Inf"):
        _core.sdm_kmeans(bad, 3, seed=1)


# as.character() of doubles in R 4.4.2 on Windows (scientific() in long double,
# printing by trio): the text by which R groups numbers in as.factor() / merge()
AS_CHARACTER = [
    ("0x1.31d8e659p+3", "9.5577270258218"),  # %.15g: 9.55772702582181
    ("0x1.0d588ab4e91a3p-32", "2.4496863852255e-10"),
    ("0x1.d47d51c23b7a1p-16", "2.79241519444622e-05"),
    ("0x1.25802c70c6a7dp+51", "2581659266725183"),
    ("0x1.20f558e39cc94p+49", "635426214197651"),
    ("0x1.3ea07a04fc3e4p+66", "91837940248660738058"),  # trio's digits past 19
    ("0x1.1aa643bc156cdp+126", "9.3926397478208e+37"),
    ("0x1.61e99c894338ap-107", "8.52013459661976e-33"),
    ("0x1.3333333333334p-2", "0.3"),
    ("0x1.5555555555555p-2", "0.333333333333333"),
    ("-0x1.5555555555555p-1", "-0.666666666666667"),
    ("0x1.4f8b588e368f1p-17", "1e-05"),
    ("0x1.86ap+16", "1e+05"),
    ("0x1.e24p+16", "123456"),
    ("0x1.c6bf52634p+49", "1e+15"),
    ("0x1.1c37937e08p+53", "1e+16"),
    ("0x1.869ffffffffffp+16", "1e+05"),
    ("0x1.4p+3", "10"),
    ("0x1.c6bf52633ffffp+49", "1e+15"),
    ("0x1.0f0cf064dd592p+73", "1e+22"),
    ("0x1.52d02c7e14af6p+76", "1e+23"),
    ("0x1.56e1fc2f8f359p-997", "1e-300"),
    ("0x1p-1074", "4.94065645841247e-324"),
    ("0x1.fffffffffffffp+1023", "1.79769313486232e+308"),
    ("-0x0p+0", "0"),
    ("0x1.9p+6", "100"),
    ("0x1.a36e2eb1c432dp-14", "1e-04"),
    ("0x1.2d6871999999ap+20", "1234567.1"),
    ("-0x1.68fcd6e9b9cb2p+5", "-45.123456789"),
    ("0x1.ddd4baa009303p+999", "1e+301"),
]


def test_as_character_matches_r():
    x = fromhex([h for h, _ in AS_CHARACTER])
    assert _core.sdm_r_as_character(x) == [s for _, s in AS_CHARACTER]
    assert _core.sdm_r_as_character(np.array([np.nan, np.inf, -np.inf])) == ["NaN", "Inf", "-Inf"]


def test_unique_rows_are_exact():
    """unique.matrix() in R 4.4.2 compares rows as asplit() lists: == per element."""
    # R: which(!duplicated(x)) is 1 3 4 5 6 (0 and -0 alike, 0.1 + 0.2 != 0.3)
    x = np.array([[0, 1], [-0.0, 1], [0.1, 1], [0.1 * (1 + 2.0**-52), 1], [0.3, 2], [0.1 + 0.2, 2], [0.1, 1]])
    assert _core.sdm_r_unique_rows(x).tolist() == [0, 2, 3, 4, 5]
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, size=(5000, 3)).astype(float)
    seen, keep = set(), []
    for i, row in enumerate(map(tuple, y)):
        if row not in seen:
            seen.add(row)
            keep.append(i)
    assert _core.sdm_r_unique_rows(y).tolist() == keep
