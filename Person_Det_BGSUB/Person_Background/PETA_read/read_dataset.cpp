#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <Windows.h>
using namespace std;
long int Total_imgs = 0;
void single_dataset_read(char **argv,const string &dataset_name)
{
	string dir_name = argv[1]+dataset_name;
	string tmp_name = "\\" +dataset_name;
	string print_dir_name = argv[1] + tmp_name;
	string lab_name = dir_name + "\\archive\\Label.txt";//Each dataset contains a Label.txt file for info
	string line;
	ifstream myfile(lab_name);//File reading
	vector<string> prefs;//All the image's prefixes that contains the attribute 
	if (myfile.is_open())
	{
		while (getline(myfile, line))
		{
			stringstream ss(line);
			string tok;
			vector<string> words;
			int i = 0;
			string num;
			while (getline(ss, tok, ' ')) {
				if (i == 0)
				{
					num = tok;
					i = 1;
				}
				words.push_back(tok);
				if (tok == argv[3])
				{
					prefs.push_back(num);
					break;
				}
			}
		}
		myfile.close();//Finished reading the file


	}
	else cout << "Unable to open file";

	/*Writing prefix of the images with the given attributes*/
#ifdef _WIN32
	ofstream out(argv[2], std::ios_base::app);

	for (size_t i = 0; i < prefs.size(); i++)
	{
		string strSearch = dir_name + "\\archive\\" + prefs[i] + "*";
		WIN32_FIND_DATAA ffd;
		HANDLE hFind = FindFirstFileA(strSearch.c_str(), &ffd);
		do
		{
			std::string strFile = ffd.cFileName;
			out << print_dir_name + "\\\\archive\\\\" + strFile << endl;
			Total_imgs++;
		} while (FindNextFileA(hFind, &ffd) != 0);
	}
	out.close();
#elif __linux__
	cout << "Not yet implemented on UNIX" << endl;
#endif
}

int main(int argc,char** argv)
{
	if (argc != 4)
	{
		cout << "Usage: " << argv[0] << "Dataset_directory_path Output_filename_with_full_path" 
			<< "attribute_name" << endl;
		cout << "Fore more information see the format of a Lebel.txt file" << endl;
		cout << "Example: PETA_read PETA dataset\\ accessoryHat.txt accessoryHat" << endl;
		return 1;
	}

	vector<string> datasets;
	datasets.push_back("3DPeS");
	datasets.push_back("CAVIAR4REID");
	datasets.push_back("CUHK");
	datasets.push_back("GRID");
	datasets.push_back("i-LID");
	datasets.push_back("MIT");
	datasets.push_back("PRID");
	datasets.push_back("SARC3D");
	datasets.push_back("TownCentre");
	datasets.push_back("VIPeR");
	string ans="N";
	for (size_t j = 0; j < datasets.size(); j++)
	{
		cout << "Do you want to read " << datasets[j] << " dataset?(Y/N)" << endl;
		cin >> ans;
		if (ans == "Y")
		{
			single_dataset_read(argv, datasets[j]);
			cout << "Selected " << datasets[j] << endl;
		}
			
		//cout << "Done" << endl;
	}
	cout << "Total number of images: " << Total_imgs << endl;
	
	return 0;
}