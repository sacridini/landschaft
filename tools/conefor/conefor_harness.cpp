// SPDX-License-Identifier: GPL-3.0-or-later
//
// Command-line driver around Conefor Sensinode 2.2's index code
// (landscape.cpp, GPL-3), used to produce reference values for landschaft's
// connectivity graphs. It replays sensinodeProgram's flow: the global
// landscape, one landscape per deleted node and one per node with area 0,
// and the dI formulas of sensinodeProgram.cpp (in float, as Conefor).
//
// Input file:
//   n adjacentTail kDistanceProb kProbProb minProb landscapeArea
//   n lines: label size
//   n lines of n distances
// Output (stdout, CSV): overall indices, then one row per node.
#include <cstdio>
#include <vector>

#include "landscape.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: conefor_harness input.txt\n");
        return 2;
    }
    FILE* f = std::fopen(argv[1], "r");
    if (!f) return 2;
    int n;
    float tail, kdist, kprob, minprob, area;
    if (std::fscanf(f, "%d %f %f %f %f %f", &n, &tail, &kdist, &kprob, &minprob, &area) != 6) return 2;
    std::vector<int> label(n);
    std::vector<float> size(n);
    for (int i = 0; i < n; ++i)
        if (std::fscanf(f, "%d %f", &label[i], &size[i]) != 2) return 2;
    std::vector<std::vector<float>> conn(n, std::vector<float>(n));
    std::vector<float*> rows(n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j)
            if (std::fscanf(f, "%f", &conn[i][j]) != 1) return 2;
        rows[i] = conn[i].data();
    }
    std::fclose(f);
    // sensinodeProgram: no landscape area -> 1 (indices = their numerators).
    const int la = area > 0 ? static_cast<int>(area) : 1;
    float total = 0;
    for (int i = 0; i < n; ++i) total = total + size[i];

    auto run = [](landscape& l) {
        l.generate_index_NL();
        l.generate_index_NC();
        l.generate_index_H_IIC();
        l.generate_index_CCP();
        l.generate_index_LCP();
        l.generate_index_F();
        l.generate_index_AWF();
        l.generate_index_PC_BASE();
    };
    landscape global;
    global.create_land(size.data(), label.data(), rows.data(), n, la, true, true, true, false, false, tail, kprob,
                       kdist, minprob, false);
    run(global);
    std::printf("overall,NC,%d\noverall,NL,%d\noverall,H,%.9g\noverall,CCP,%.9g\noverall,LCP,%.9g\n",
                global.value_index_NC(), global.value_index_NL(), global.value_index_H(), global.value_index_CCP(),
                global.value_index_LCP());
    std::printf("overall,LCPnum,%.9g\noverall,IIC,%.9g\noverall,IICnum,%.9g\noverall,F,%.9g\noverall,AWF,%.9g\n",
                global.value_index_LCPnum(), global.value_index_IIC(), global.value_index_IICnum(),
                global.value_index_F(), global.value_index_AWF());
    std::printf("overall,PC,%.9g\noverall,PCnum,%.9g\n", global.value_index_PC(), global.value_index_PCnum());

    std::printf("node,label,dA,dNC,dNL,dH,dCCP,dLCP,dIIC,dF,dAWF,dPC,dPCintra,dPCflux,dPCconnector,varPC\n");
    for (int i = 0; i < n; ++i) {
        landscape del, zero;
        del.create_land_with_copy_deleting_one(&global, i, true, true);
        run(del);
        zero.create_land_with_copy_area0_one_node(&global, i, true, true);
        zero.generate_index_PC_BASE();
        const float dA = 100 * size[i] / total;
        const float dNC = 100.0 * (global.value_index_NC() - del.value_index_NC()) / global.value_index_NC();
        float dNL = 0;
        if (global.value_index_NL() != 0)
            dNL = 100.0 * (global.value_index_NL() - del.value_index_NL()) / global.value_index_NL();
        float dH = 0;
        if (global.value_index_H() != 0)
            dH = 100 * (global.value_index_H() - del.value_index_H()) / global.value_index_H();
        const float dCCP = 100 * (global.value_index_CCP() - del.value_index_CCP()) / global.value_index_CCP();
        const float dLCP = 100 * (global.value_index_LCP() - del.value_index_LCP()) / global.value_index_LCP();
        const float dIIC = 100 * (global.value_index_IIC() - del.value_index_IIC()) / global.value_index_IIC();
        const float dF = 100 * (global.value_index_F() - del.value_index_F()) / global.value_index_F();
        const float dAWF = 100 * (global.value_index_AWF() - del.value_index_AWF()) / global.value_index_AWF();
        const float dPC = 100 * (global.value_index_PC() - del.value_index_PC()) / global.value_index_PC();
        const float dPCintra = 100 * ((size[i] / la) * (size[i] / la)) / global.value_index_PC();
        const float dPCconnector = 100 * (zero.value_index_PC() - del.value_index_PC()) / global.value_index_PC();
        const float dPCflux = dPC - dPCintra - dPCconnector;
        const float varPC = global.value_index_PC() - del.value_index_PC();
        std::printf("node,%d,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g\n", label[i], dA, dNC,
                    dNL, dH, dCCP, dLCP, dIIC, dF, dAWF, dPC, dPCintra, dPCflux, dPCconnector, varPC);
    }
    return 0;
}
