#pragma once
#include <string>
#include <vector>
#include <math.h>
#define NN_NUM 5

const double Epsilon = 1e-10;

class Recognizer {
private:
    const std::string working_path; 
    //struct fann *ann;
    const double threshold1;
    struct fann* ann[NN_NUM];
    struct fann* top_ann;
    int dd;
    std::vector<int> hiddens;
public:
    Recognizer(const std::string& path, double thrd)
	: working_path(path)
	, threshold1(thrd)
	, top_ann(nullptr)
    {
	for (int i = 0; i < NN_NUM; ++i) {
	    ann[i] = nullptr;
	}
    }

    ~Recognizer()
    {}

    void preprocess(const std::string& imagef, const std::string& labelf); 

    //void* train_dd(void* nothing);

    void train_nn();

    void prepare_top_train(const std::string& imagef, const std::string& labelf);

    void train_top_nn();

    int classification(unsigned char* inputs) const;

    int max(int r, int l) const  {return abs(r)> abs(l)?abs(r):abs(l);}

    void Kirsch(unsigned char *image, int row, int column, int direction, int* features) const;

    void compress(int* old, int row, int column, double* now) const;

    int sixteen_into_one(int* old, int column, int min_row_index, int min_column_index) const;

    void features_extraction(unsigned char *image, int row, int column, int direction, double* features) const;

    void load_nn();
    
    void load_nn_all();

    void test(const std::string& imagef, const std::string& ilabelf);

    void canny(unsigned char *image, int row, int column, unsigned char* new_image) const;

    void contour_descriptors(unsigned char *image, int row, int column, double* features) const;
    
    void my_descriptors(unsigned char *image, int row, int column, double* features) const;

    double sigmod(double v) const {
	const double d =  1 / (1+exp(-v));
	return d > Epsilon?d:0 ;
    }
};
