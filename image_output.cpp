#include <iostream>
#include <fstream>
#include <string>

using std::ofstream;
using std::string;
using std::ios;

// Writes contents of imageBytes to a ppm file is active directory
// Imagebytes MUST be at leasy xRes*yRes*3 bytes long
void writePPM(const unsigned int xRes, const unsigned int yRes, const char* imageBytes, const string fileName) {
    if (imageBytes == NULL) {
        std::cout << "Error writing PPM: cannot read from NULL ptr";
        return;
    }

    ofstream outFile;
    if (fileName.find(".ppm") != string::npos) { // Appends ".ppm" to the filename if the extension is not already present
        outFile.open(fileName, ios::binary);
    }
    else {
        outFile.open(fileName + ".ppm", ios::binary);
    }

    // Checks if file opened succesfully;
    if (outFile) {
    outFile << "P6\n" << xRes << " " << yRes << "\n255\n"; // Header to image, P6 is the code for a binary-encoded colour image, 255 is the width of each colour channel
    outFile.write(imageBytes, xRes * yRes * sizeof(unsigned char) * 3);
    outFile.close();
    } else {
        std::cout << "Error writing PGM: Could not open file: " + fileName;
    }
}

// As above, but for a greyscale pgm image instead
// Imagebytes MUST be at leasy xRes*yRes bytes long
void writePGM(const unsigned int xRes, const unsigned int yRes, const char* imageBytes, const string fileName) {
    if (imageBytes == NULL) {
        std::cout << "Error writing PGM: cannot read from NULL ptr";
        return;
    }

    ofstream outFile;
    if (fileName.find(".pgm") != string::npos) { // Appends ".pgm" to the filename if the extension is not already present
        outFile.open(fileName, ios::binary);
    }
    else {
        outFile.open(fileName + ".pgm", ios::binary);
    }

    // Checks if file opened succesfully;
    if (outFile) {
        outFile << "P5\n" << xRes << " " << yRes << "\n255\n"; // P5 for binary-encoded greyscale, 255 is max value of each pixel (I.E. a white pixel)
        outFile.write(imageBytes, xRes * yRes * sizeof(unsigned char));
        outFile.close();
    } else {
        std::cout << "Error writing PGM: Could not open file: " + fileName;
    }
}