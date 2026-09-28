#!/usr/bin/perl
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Run the original Biodiverse on a job description and dump the results as JSON.
#
#   perl -I<biodiverse lib> tools/bio_reference.pl job.json out.json
#
# Driven by tools/bio_harness.py, which also builds the sorted-order overlay of the
# Biodiverse library (see there). Numbers are written with 17 significant digits,
# so that Python reads back the same doubles.

use strict;
use warnings;
use 5.022;

use Carp;
use Cpanel::JSON::XS ();
use Scalar::Util qw /looks_like_number blessed reftype/;

use Biodiverse::BaseData;
use Biodiverse::ReadNexus;
use Biodiverse::Tree;
use Biodiverse::Indices;
use Biodiverse::Matrix;

local $| = 1;

my ($job_file, $out_file) = @ARGV;
croak "usage: bio_reference.pl job.json out.json\n" if !defined $out_file;

my $job = do {
    open my $fh, '<:raw', $job_file or croak "cannot open $job_file";
    local $/;
    Cpanel::JSON::XS->new->utf8->decode(<$fh>);
};

my $bd     = load_basedata($job->{basedata}) if $job->{basedata};
my $tree   = load_tree($job->{tree})         if $job->{tree};
my $matrix = load_matrix($job->{matrix})     if $job->{matrix};

my %output;
foreach my $task (@{$job->{tasks} // []}) {
    my $type = $task->{type};
    my $name = $task->{name} // $type;
    my $result
      = $type eq 'basedata' ? dump_basedata($bd)
      : $type eq 'tree'     ? dump_tree($tree)
      : $type eq 'trees'    ? dump_trees($task)
      : $type eq 'matrix'   ? dump_matrix($matrix)
      : $type eq 'indices'  ? run_indices($bd, $tree, $matrix, $task)
      : $type eq 'spatial'  ? run_spatial($bd, $tree, $matrix, $task)
      : $type eq 'neighbours' ? run_neighbours($bd, $task)
      : croak "unknown task type $type";
    $output{$name} = $result;
}

open my $ofh, '>:raw', $out_file or croak "cannot write $out_file";
print {$ofh} encode_value(\%output);
close $ofh;
exit 0;


sub load_basedata {
    my ($spec) = @_;
    my %spec = %$spec;
    my $file   = delete $spec{file};
    my $raster = delete $spec{raster_files};
    my $shapes = delete $spec{shapefiles};
    my $cell_sizes = delete $spec{cell_sizes};
    my $bd = Biodiverse::BaseData->new(
        NAME         => 'landschaft reference',
        (defined $cell_sizes ? (CELL_SIZES => $cell_sizes) : ()),
        (defined $spec{cell_origins} ? (CELL_ORIGINS => delete $spec{cell_origins}) : ()),
    );
    delete $spec{cell_origins};
    if ($raster) {
        $bd->import_data_raster(input_files => $raster, %spec);
    }
    elsif ($shapes) {
        $bd->import_data_shapefile(input_files => $shapes, %spec);
    }
    else {
        $bd->import_data(input_files => [$file], %spec);
    }
    return $bd;
}

sub load_tree {
    my ($spec) = @_;
    my $read_nex = Biodiverse::ReadNexus->new;
    my %args = defined $spec->{file} ? (file => $spec->{file}) : (data => $spec->{data});
    $read_nex->import_data(%args);
    my @trees = $read_nex->get_tree_array;
    return $trees[$spec->{index} // 0];
}

sub load_matrix {
    my ($spec) = @_;
    my $matrix = Biodiverse::Matrix->new(
        NAME => 'landschaft reference',
        %{$spec->{params} // {}},
    );
    if ($spec->{sparse}) {
        $matrix->import_data_sparse(file => $spec->{file}, %{$spec->{args} // {}});
    }
    else {
        $matrix->import_data(file => $spec->{file}, %{$spec->{args} // {}});
    }
    return $matrix;
}

sub dump_basedata {
    my ($bd) = @_;
    my $gp_ref = $bd->get_groups_ref;
    my %groups;
    foreach my $gp (sort $bd->get_groups) {
        $groups{$gp} = {
            coord   => [ $gp_ref->get_element_name_as_array(element => $gp) ],
            labels  => { %{ $bd->get_labels_in_group_as_hash_aa($gp) } },
        };
    }
    my %labels;
    foreach my $lb (sort $bd->get_labels) {
        $labels{$lb} = {
            range        => $bd->get_range(element => $lb),
            sample_count => $bd->get_label_sample_count(element => $lb),
        };
    }
    return {
        groups         => \%groups,
        labels         => \%labels,
        cell_sizes     => [ $bd->get_cell_sizes ],
        cell_origins   => [ $bd->get_cell_origins ],
        numeric_labels => $bd->get_param('NUMERIC_LABELS'),
    };
}

sub dump_tree {
    my ($tree) = @_;
    my %nodes;
    foreach my $node ($tree->get_node_refs) {
        my $parent = $node->get_parent;
        $nodes{$node->get_name} = {
            length   => $node->get_length,
            parent   => ($parent ? $parent->get_name : undef),
            terminal => ($node->is_terminal_node ? 1 : 0),
            children => [ map { $_->get_name } $node->get_children ],
        };
    }
    return {
        nodes        => \%nodes,
        total_length => $tree->get_total_tree_length,
    };
}

sub dump_trees {
    my ($task) = @_;
    my $read_nex = Biodiverse::ReadNexus->new;
    my %args = defined $task->{file} ? (file => $task->{file}) : (data => $task->{data});
    $read_nex->import_data(%args);
    return [ map { { name => $_->get_param('NAME'), %{ dump_tree($_) } } } $read_nex->get_tree_array ];
}

sub dump_matrix {
    my ($matrix) = @_;
    my @elements = sort $matrix->get_elements_as_array;
    my %pairs;
    foreach my $e1 (@elements) {
        foreach my $e2 (@elements) {
            my $v = $matrix->get_defined_value_aa($e1, $e2);
            $pairs{$e1}{$e2} = $v if defined $v;
        }
    }
    return { elements => \@elements, pairs => \%pairs };
}

sub calc_args {
    my ($tree, $matrix, $task) = @_;
    return (
        ($tree   ? (tree_ref   => $tree)   : ()),
        ($matrix ? (matrix_ref => $matrix) : ()),
        %{ $task->{args} // {} },
    );
}

#  Indices on given neighbour sets, like the Biodiverse test helpers.
sub run_indices {
    my ($bd, $tree, $matrix, $task) = @_;
    my %calc_args = calc_args($tree, $matrix, $task);
    my %elements = (element_list1 => $task->{element_list1});
    $elements{element_list2} = $task->{element_list2}
      if defined $task->{element_list2};
    my $nbr_list_count = defined $task->{element_list2} ? 2 : 1;

    my $indices = Biodiverse::Indices->new(BASEDATA_REF => $bd);
    $indices->get_valid_calculations(
        calculations   => $task->{calculations},
        nbr_list_count => $nbr_list_count,
        calc_args      => { %calc_args, %elements },
    );
    $indices->run_precalc_globals(%calc_args);
    my %results = $indices->run_calculations(%calc_args, %elements);
    $indices->run_postcalc_globals(%calc_args);
    return \%results;
}

sub run_spatial {
    my ($bd, $tree, $matrix, $task) = @_;
    my $name = $task->{name} // 'spatial';
    my $sp = $bd->add_spatial_output(name => $name);
    $sp->run_analysis(
        spatial_conditions => $task->{spatial_conditions},
        (defined $task->{definition_query} ? (definition_query => $task->{definition_query}) : ()),
        calculations => $task->{calculations},
        calc_args($tree, $matrix, $task),
    );
    my %results;
    my $pass = $sp->get_pass_def_query;
    foreach my $gp (sort $sp->get_element_list) {
        next if $pass && !$pass->{$gp};
        my %lists;
        foreach my $list (@{ $sp->get_lists(element => $gp) }) {
            next if $list =~ /^_/ || $list eq 'RESULTS_SAME_AS';
            my $ref = $sp->get_list_ref(element => $gp, list => $list);
            if ($list eq 'SPATIAL_RESULTS') {
                $ref = { %$ref };
                delete $ref->{RECYCLED_SET};
            }
            $lists{$list} = $ref;
        }
        $results{$gp} = \%lists;
    }
    $bd->delete_output(output => $sp) if !$task->{keep};
    return \%results;
}


#  Neighbour lists (_NBR_SET1, _NBR_SET2) of the groups a spatial analysis
#  processed; a group that failed the definition query is absent.
sub run_neighbours {
    my ($bd, $task) = @_;
    my $name = $task->{name} // 'neighbours';
    my $sp = $bd->add_spatial_output(name => $name);
    my $conditions = $task->{spatial_conditions};
    $sp->run_analysis(
        spatial_conditions => $conditions,
        (defined $task->{definition_query} ? (definition_query => $task->{definition_query}) : ()),
        calculations => ['calc_richness'],
        ($task->{no_recycling} ? (no_recycling => 1) : ()),
    );
    my %results;
    my $pass = $sp->get_pass_def_query;
    foreach my $gp (sort $sp->get_element_list) {
        next if $pass && !$pass->{$gp};
        my %sets;
        foreach my $i (1 .. scalar @$conditions) {
            my $l = $sp->get_list_ref(element => $gp, list => "_NBR_SET$i", autovivify => 0);
            $sets{$i} = [sort @{$l // []}];
        }
        $results{$gp} = \%sets;
    }
    $bd->delete_output(output => $sp);
    return \%results;
}

#  JSON with 17 significant digits for numbers and sorted keys.
sub encode_value {
    my ($v) = @_;
    return 'null' if !defined $v;
    my $rt = reftype $v;
    if ($rt) {
        if ($rt eq 'HASH') {
            return '{' . join(',', map { encode_string($_) . ':' . encode_value($v->{$_}) } sort keys %$v) . '}';
        }
        if ($rt eq 'ARRAY') {
            return '[' . join(',', map { encode_value($_) } @$v) . ']';
        }
        if ($rt eq 'SCALAR') {
            return encode_value($$v);
        }
        return encode_string("$v");
    }
    if (looks_like_number($v) && $v !~ /^\s|\s$/) {
        return '"Inf"'  if $v ==  9**9**9;
        return '"-Inf"' if $v == -9**9**9;
        return '"NaN"'  if $v != $v;
        return sprintf '%.17g', $v;
    }
    return encode_string($v);
}

my $json_string_encoder;
sub encode_string {
    $json_string_encoder //= Cpanel::JSON::XS->new->utf8->allow_nonref;
    return $json_string_encoder->encode("$_[0]");
}
