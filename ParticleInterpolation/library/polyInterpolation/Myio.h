#include<iostream>
#include<fstream>
#include<vector>
#include<string>

//////////////////////////////////////
// How to use?
// in put a vector from a txt file:
// input(file, data);
//
// output a txt file from a vector:
// output(file, data);
// 
/////////////////////////////////////

void input(std::string filename, std::vector<double> &data){
    if (data.size() != 0) std::cout << "the destination isn't empty!" << std::endl;
    std::cout << "Writing a vector to \"" << filename <<"\"."<< std::endl;
    std::ifstream  ifs(filename, std::ios::in);
    while(ifs.eof()==0){
        double temp;
        ifs >> temp;
        data.push_back(temp);
    }
    data.pop_back();
    ifs.close();
}

void output(std::string filename, std::vector<double> &data){
    std::cout << "Reading a vector from \"" << filename <<"\"."<< std::endl; 
    std::ofstream  ofs(filename, std::ios::out);
    for(size_t i = 0; i < data.size(); i++)
        ofs << data[i] << std::endl;
    ofs.close();
}
