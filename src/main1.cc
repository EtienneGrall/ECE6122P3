#include <iostream>
#include "complex.cc"
#include "input_image.cc"

//test your code here

int main() {
	InputImage image("../Tower1024.txt");
    std::cout << image.get_width() << std::endl;
    std::cout << image.get_height() << std::endl;
    return 0;
}