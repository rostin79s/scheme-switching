#include <fstream>
#include <vector>
#include <sstream>
#include <iostream>

void addheaders(std::string filename) {
    std::ifstream inFile(filename);
    if (!inFile) {
        std::cerr << "Error opening input file" << std::endl;
        return;
    }

    // Read all content into a string
    std::stringstream buffer;
    buffer << inFile.rdbuf();
    std::string content = buffer.str();
    inFile.close();

    // Headers to add
    std::vector<std::string> headers = {
        "#include \"src/backend/fhe_operations.hpp\"",
        "#include \"src/backend/fhe_types.hpp\"",
        "#include <vector>",
        "#include <iostream>",
        "using namespace CKKS;",
        "using namespace CGGI;",
        ""  // Empty line after headers
    };

    // Open the file for writing
    std::ofstream outFile("output.cpp");
    if (!outFile) {
        std::cerr << "Error opening output file" << std::endl;
        return;
    }

    // Write headers
    for (const auto& header : headers) {
        outFile << header << std::endl;
    }

    // Write original content
    outFile << content;
    outFile.close();

    std::cout << "Headers successfully added to output.cpp" << std::endl;
}

int main(){
    addheaders("output.cpp");
    return 0;
}