// SPDX-License-Identifier: GPL-3.0-or-later
//
// The state of a SWAT+ simulation: the variables of the SWAT+ 62.0.1 Fortran modules
// (modules.hpp, generated from them), the project's files and what the run records.

#pragma once

#include <string>
#include <vector>

#include "swat/fio.hpp"
#include "swat/modules.hpp"

namespace swat {

// One daily dump record, the same as the reference's ls_dump_module writes.
struct DumpRecord {
    int tag, yrc, day, id;
    std::vector<float> values;
};

struct Sim : input_file_module,
             time_module,
             basin_module,
             climate_module,
             maximum_data_module,
             hydrograph_module,
             plant_data_module,
             landuse_data_module,
             mgt_operations_module,
             tillage_data_module,
             fertilizer_data_module,
             pesticide_data_module,
             pathogen_data_module,
             urban_data_module,
             septic_data_module,
             soil_data_module,
             soil_module,
             plant_module,
             carbon_module,
             organic_mineral_mass_module,
             hru_module,
             hydrology_data_module,
             topography_data_module,
             output_landscape_module,
             aquifer_module,
             sd_channel_module,
             channel_velocity_module,
             channel_data_module,
             channel_module,
             ru_module,
             calibration_data_module,
             conditional_module,
             water_body_module,
             erosion_module,
             constituent_mass_module,
             tiles_data_module,
             reservoir_data_module,
             reservoir_module,
             exco_module,
             dr_module,
             recall_module {
    fio::FileSet files;

    // what the reference writes to diagnostics.out
    std::vector<std::string> warnings;
    void warn(const std::string& m) { warnings.push_back(m); }

    // daily dump
    bool dump = false;
    std::vector<DumpRecord> dumps;
    void dump_record(int tag, int id, std::vector<float> v) {
        if (dump) dumps.push_back({tag, time.yrc, time.day, id, std::move(v)});
    }
};

}  // namespace swat
