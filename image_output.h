#pragma once

#include <string>

// Writes a ppm image file at given resolution, from array located at imageBytes, with given filename
// Imagebytes MUST be at leasy xRes*yRes*3 bytes long
void writePPM(const unsigned int xRes, const unsigned int yRes, const char* imageBytes, const std::string fileName);

// ^^ but with a pgm instead
// Imagebytes MUST be at leasy xRes*yRes bytes long
void writePGM(const unsigned int xRes, const unsigned int yRes, const char* imageBytes, const std::string fileName);