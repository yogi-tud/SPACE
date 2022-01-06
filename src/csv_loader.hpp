#pragma once

#include <deps/csv.hpp>
#include <stdlib.h>
#include <iterator>

typedef std::initializer_list<int>::iterator ilii;
void append_to_each(csv::CSVRow& row, ilii idxs, ilii idxs_end)
{
    UNUSED(row);
    UNUSED(idxs);
    UNUSED(idxs_end);
}

template <typename COL1, typename... COL_TYPES>
void append_to_each(csv::CSVRow& row, ilii idxs, ilii idxs_end, std::vector<COL1>& col1, std::vector<COL_TYPES>&... cols)
{
    col1.push_back(row[*idxs].get<COL1>());
    append_to_each(row, ++idxs, idxs_end, cols...);
}

template <typename... COL_TYPES> int load_csv(const char* path, std::initializer_list<int> column_indices, std::vector<COL_TYPES>&... cols)
{
    try {
        csv::CSVFormat format;
        format.no_header().delimiter('|');
        csv::CSVReader reader(path, format);
        for (csv::CSVRow& row : reader) {
            append_to_each(row, column_indices.begin(), column_indices.end(), cols...);
        }
    }
    catch (const std::exception& ex) {
        std::cout << "exception: " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
    catch (...) {
        std::cout << "fatal error" << std::endl;
        return EXIT_SUCCESS;
    }
    return 0;
}
