#include "text_display.cuh"
#include <stdlib.h>
#include <stdio.h>
#include <string>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>


// This function is windows-specific. Similar functionality could probably be achieved using ANSI escape codes on 
// non-windows systems, but unfortunatly I am not able to test that on my system. 
void ClearScreen()
{
    HANDLE                     hStdOut;
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    DWORD                      count;
    DWORD                      cellCount;
    COORD                      homeCoords = { 0, 0 };

    hStdOut = GetStdHandle(STD_OUTPUT_HANDLE);
    if (hStdOut == INVALID_HANDLE_VALUE) return;

    /* Get the number of cells in the current buffer */
    if (!GetConsoleScreenBufferInfo(hStdOut, &csbi)) return;
    cellCount = csbi.dwSize.X * csbi.dwSize.Y;

    /* Fill the entire buffer with spaces */
    if (!FillConsoleOutputCharacter(
        hStdOut,
        (TCHAR)' ',
        cellCount,
        homeCoords,
        &count
    )) return;

    /* Fill the entire buffer with the current colors and attributes */
    if (!FillConsoleOutputAttribute(
        hStdOut,
        csbi.wAttributes,
        cellCount,
        homeCoords,
        &count
    )) return;

    /* Move the cursor home */
    SetConsoleCursorPosition(hStdOut, homeCoords);
}
#endif


// Used to map an intensity/brightness value to an ascii character. Characters get brighter as you move to the right
const std::string ramp = "  .\":-=+-?X*#&%@";


unsigned char asciiFromIntensity(unsigned char intensity) {
    return ramp[intensity * ramp.length() / 256]; // Scales our unsigned char to be in the rang [0, ramp.length] and indexes ramp
}


// Simple text output for greyscale bitmap. Assumes range of 0-255. All downscaling/upscaling is done using a single sample
void displayGreyscale(unsigned char* pixels, unsigned int imageWidth, unsigned int imageHeight, unsigned int consoleWidth, unsigned int consoleHeight) {
    ClearScreen();
	unsigned char* outputBuffer = (unsigned char*)malloc(consoleWidth * sizeof(unsigned char) + 1);
    
    outputBuffer[consoleWidth] = '\0';
    printf("Current frame:\n");
    printf("Character ramp for display: %s\n", ramp);
    for (unsigned int col = 0; col < consoleHeight; col++) {
        for (unsigned int row = 0; row < consoleWidth; row++) {
            // Index's the appropriate pixel from the image for each character in the console output, and adds a character to the buffer
            outputBuffer[row] = asciiFromIntensity(pixels[row * imageWidth / consoleWidth   +   col * imageHeight / consoleHeight * imageWidth]);
        }
        // Prints our buffer to the screen before moving on to the next row
        printf("%s\n", (char*)outputBuffer);
    }
    free(outputBuffer);
}