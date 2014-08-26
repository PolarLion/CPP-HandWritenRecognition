#include "Recognizer.h"
//#include "fann.h"
#include "doublefann.h"
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <cmath>
#include <ctime>
#include <chrono>
#include <stdlib.h>
#include <sstream>
#include <pthread.h>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

const double  Pi = 3.1415926;
const int training_start = 0;

using namespace std;
using namespace cv;

int number = training_start;
void Recognizer::preprocess(const string& imagef, const string& labelf)
{
    unordered_map<int, string> map0;
    string l = "0 0 0 0 0 0 0 0 0 0";
    for (int i = 0; i < 10; ++i) {
	string tl = l;
	tl[2 * i] = '1';
	map0[i] = tl;
    }
    ifstream iimagef(imagef);
    if ( iimagef.fail() ) {
	printf ("open file error %s\n", imagef.c_str());
	return;
    }
    ifstream ilabelf(labelf);
    if ( ilabelf.fail() ) {
	printf ("open file error %s\n", labelf.c_str());
	return;
    }
    /*ofstream outfile(working_path+"train.txt");
      if ( outfile.fail() ) {
      printf ("open outfile error %s]n", (working_path+"train.txt").c_str());
      return;
      }*/

    char bit_32[4];
    //read MSB
    iimagef.read(bit_32, 4);
    ilabelf.read(bit_32, 4);
    //read number of items
    iimagef.read(bit_32, 4);
    ilabelf.read(bit_32, 4);
    const int size = 60000;
    //read nuber of rows (only in images)
    iimagef.read(bit_32, 4);
    const int rows = 28;
    //read number of columns (only in images)
    iimagef.read(bit_32, 4);
    const int columns = 28;
    const int inputs_d = rows * columns;
    //cout << "bit_32 : " <<(int)bit_32[3] << endl;
    //outfile << size << " " << inputs_d << " " << 10 << endl;
    char byte[0];
    int count = 0;
    int count_byte = 0;
    unsigned char *a_image = new unsigned char[inputs_d];
    const int n = rows * columns / 16;
    for (int i = training_start; i < NN_NUM; ++i ) {
	stringstream ss;
	ss << i;
	string filename = working_path + "train" + ss.str() + ".txt";
	ofstream outfile(filename);
	outfile << 60000 << " " << 49 << " " << 10 << endl;
	outfile.close();
    }
    while ( !iimagef.eof() ) {
	for ( int i = 0; i < inputs_d; ++i ) {
	    //iimagef >> ubyte;
	    iimagef.read(byte, 1);
	    if ( iimagef.eof() )
		break;
	    a_image[i] = (unsigned char)byte[0];
	    //++count_byte;
	    //unsigned char ubyte = (unsigned char)byte[0];
	    //outfile << (int)ubyte / (double)255 << " ";
	}
	ilabelf.read(byte, 1);
	for (int i = training_start; i < NN_NUM; ++i ) {
	    stringstream ss;
	    ss << i;
	    string filename = working_path + "train" + ss.str() + ".txt";
	    ofstream outfile(filename, ios::app);
	    double* inputs = new double[n];
	    features_extraction(a_image, rows, columns, i, inputs);
	    for (int ii = 0; ii < n; ++ii) {
		outfile << inputs[ii] << " ";// / (double)3825 << " ";
	    }
	    outfile << endl;
	    //
	    outfile << map0[byte[0]] << endl;
	    outfile.close();
	    delete inputs;
	}
	//++count;
	//outfile << endl << map0[byte[0]] << endl;
    }
    delete a_image;
    //cout << count << endl;
    //cout << count_byte << endl;

    //outfile.close();
    iimagef.close();
    ilabelf.close();
}


void* train_flnn(void* nothing)
{
    stringstream ss;
    ss << number++;
    const unsigned int num_layers = 3;
    const float bitlimit = 20.0;
    const unsigned int max_epochs = 1000;
    const unsigned int epochs_between_reports = 100;
    struct fann* nn;
    //stringstream ss;
    //ss << dd;
    string train_file = "io/train" + ss.str() + ".txt";
    printf("read train file %s\n", train_file.c_str());
    fann_train_data *data = fann_read_train_from_file(train_file.c_str());
    //const int hidden_num = (int)pow(data->num_input * data->num_output, 0.5);
    //const int hidden_num = data->num_input;
    const int hidden_num = 33;
    //hiddens.push_back(hidden_num);
    //const int hidden_num2 = 64;
    //hiddens.push_back(hidden_num2);
    cout << "hidden num " << hidden_num << endl;
    printf("Creating network\n");
    nn = fann_create_standard(num_layers, data->num_input, hidden_num, data->num_output);
    printf("%d %d %d\n", data->num_data, data->num_input, data->num_output);
    printf("finish creating\n");
    fann_set_activation_function_hidden(nn, FANN_SIGMOID);
    fann_set_activation_function_output(nn, FANN_SIGMOID);
    fann_set_training_algorithm(nn, FANN_TRAIN_RPROP);
    fann_set_train_stop_function(nn, FANN_STOPFUNC_BIT);
    fann_init_weights(nn, data);
    fann_train_on_data(nn, data, max_epochs, epochs_between_reports, bitlimit);

    fann_save(nn, ("io/nn" + ss.str() + ".net").c_str());
    fann_destroy_train(data);
    fann_destroy(nn);
    nn = nullptr;
    return NULL;
}


void Recognizer::train_nn()
{
    pthread_t p[5];
    void *tret;
    for (int i = training_start; i < NN_NUM; ++i) {
	//number = i;
	int err = pthread_create(&p[i], NULL, train_flnn, NULL);
	if ( 0 != err ) {
	    printf ("pthread_create error:\n");
	    return;
	}
	//err = pthread_join(p[i], &tret);
    }
    for ( int i = training_start; i < NN_NUM; ++i ) {
	int err = pthread_join(p[i], &tret);
    }
}


void Recognizer::prepare_top_train(const string& imagef, const string& labelf)
{
    unordered_map<int, string> map0;
    string l = "0 0 0 0 0 0 0 0 0 0";
    for (int i = 0; i < 10; ++i) {
	string tl = l;
	tl[2 * i] = '1';
	map0[i] = tl;
    }

    ifstream iimagef(imagef);
    if ( iimagef.fail() ) {
	printf ("open file error %s\n", imagef.c_str());
	return;
    }
    ifstream ilabelf(labelf);
    if ( ilabelf.fail() ) {
	printf ("open file error %s\n", labelf.c_str());
	return;
    }

    char bit_32[4];
    //read MSB
    iimagef.read(bit_32, 4);
    ilabelf.read(bit_32, 4);
    //read number of items
    iimagef.read(bit_32, 4);
    ilabelf.read(bit_32, 4);
    const int size = 60000;
    //read nuber of rows (only in images)
    iimagef.read(bit_32, 4);
    const int rows = 28;
    //read number of columns (only in images)
    iimagef.read(bit_32, 4);
    const int columns = 28;
    const int inputs_d = rows * columns;
    cout << "bit_32 : " <<(int)bit_32[3] << endl;
    char byte[1];
    int count_byte = 0;
    unsigned char image[784];
    ofstream outfile(working_path+"top_nn.txt");
    if ( outfile.fail() ) {
	printf ("prepare_top_train : error in opening outfile\n");
	return;
    }
    outfile << 60000 << " " << 50 << " " << 10 << endl;
    double res[10];
    int count = 0;
    while ( !iimagef.eof() ) {
	for ( int i = 0; i < inputs_d; ++i ) {
	    //iimagef >> ubyte;
	    iimagef.read(byte, 1);
	    if ( iimagef.eof() )
		break;
	    ++count_byte;
	    unsigned char ubyte = (unsigned char)byte[0];
	    image[i] = ubyte;
	    //cout << (int)image[i] << " ";
	}
	//cout << endl;
	for (int i = 0; i < NN_NUM; ++i ) {
	    double inputs[49];
	    //int num = 0;
	    //Kirsch(image, 20, 20, i, inputs, num);
	    features_extraction(image, 28, 28, i, inputs);
	    /*for (int ii = 0; ii < 49; ++ii ) {
		cout << inputs[ii] << " " ;
	    }*/
	    //cout << endl;
	    fann_type *out = fann_run(ann[i], inputs);
	    for ( int ii = 0; ii < 10; ++ii ) {
		outfile << out[ii] << " ";
	    }
	    //break;
	}
	ilabelf.read(byte, 1);
	outfile << endl << map0[byte[0]] << endl;
	count++;
	//break;
    }
    //cout << "? " << count << endl;
}


void Recognizer::train_top_nn()
{
    const unsigned int num_layers = 3;
    const float bitlimit = 0;
    const unsigned int max_epochs = 1000;
    const unsigned int epochs_between_reports = 100;
    struct fann* nn;
    //stringstream ss;
    //ss << dd;
    string train_file = working_path+"top_nn.txt";
    printf("read train file %s\n", train_file.c_str());
    fann_train_data *data = fann_read_train_from_file(train_file.c_str());
    //const int hidden_num = (int)pow(data->num_input * data->num_output, 0.5);
    const int hidden_num = 33;
    //const int hidden_num2 = 10;
    cout << "hidden num " << hidden_num << endl;
    printf("Creating network\n");
    nn = fann_create_standard(num_layers, data->num_input, hidden_num, data->num_output);
    printf("%d %d %d\n", data->num_data, data->num_input, data->num_output);
    printf("finish creating\n");
    fann_set_activation_function_hidden(nn, FANN_SIGMOID);
    fann_set_activation_function_output(nn, FANN_SIGMOID);
    fann_set_training_algorithm(nn, FANN_TRAIN_RPROP);
    fann_set_train_stop_function(nn, FANN_STOPFUNC_BIT);
    fann_init_weights(nn, data);
    fann_train_on_data(nn, data, max_epochs, epochs_between_reports, bitlimit);

    fann_save(nn, (working_path + "top.net").c_str());
    fann_destroy_train(data);
    fann_destroy(nn);
    nn = nullptr;

}


void Recognizer::Kirsch(unsigned char* image, int row, int column, int direction, int* features) const 
{
    if ( nullptr == features ) {
	printf ("Kirsch : can't allocate memory\n");
	return;
    }

    if ( 0 == direction) {
	for (int i = 0; i < row; ++i ) {
	    for ( int j = 0; j < column; ++j) {
		//cout << "i = " << i << ", j = " << j << " : ";
		int k0 = 0, k4 = 0;
		if (i - 1 >= 0) {
		    if ( j - 1 >= 0 ) {
			k0 += 5 * image[(i-1)*column + j-1];
			k4 += -3 * image[(i-1)*column + j-1];
		    }
		    if ( j + 1 < column ) {
			k0 += 5 * image[(i-1)*column + j+1];
			k4 += -3 * image[(i-1)*column + j+1];
		    }
		    k0 += 5 * image[(i-1)*column + j];
		    k4 += -3 * image[(i-1)*column + j];
		}
		if ( i + 1 < row ) {
		    if ( j - 1 >= 0 ) {
			k0 += -3 * image[(i+1)*column + j-1];
			k4 += 5 * image[(i+1)*column + j-1];
		    }
		    if ( j + 1 < column) {
			k0 += -3 * image[(i+1)*column + j+1];
			k4 += 5 * image[(i+1)*column + j+1];
		    }
		    k0 += -3 * image[(i+1)*column + j];
		    k4 += 5 * image[(i+1)*column + j];
		}
		if ( j - 1 >= 0 ) {
		    k0 += -3 * image[i * column + j-1];
		    k4 += -3 * image[i * column + j-1];
		}
		if ( j + 1 < column) {
		    k0 += -3 * image[i * column + j+1];
		    k4 += -3 * image[i * column + j+1];
		}
		features[i*column +j] = max(k0, k4);
		//cout << "k0 = " << k0 << ", k4 = " << k4 << ", max = " << max(k0, k4) << endl;
	    }
	}
    }
    else if ( 1 == direction ) {
	for (int i = 0; i < row; ++i ) {
	    for ( int j = 0; j < column; ++j) {
		//cout << "i = " << i << ", j = " << j << " : ";
		int k0 = 0, k4 = 0;
		if (i - 1 >= 0) {
		    if ( j - 1 >= 0 ) {
			k0 += 5 * image[(i-1)*column + j-1];
			k4 += -3 * image[(i-1)*column + j-1];
		    }
		    if ( j + 1 < column ) {
			k0 += -3 * image[(i-1)*column + j+1];
			k4 += 5 * image[(i-1)*column + j+1];
		    }
		    k0 += -3 * image[(i-1)*column + j];
		    k4 += -3 * image[(i-1)*column + j];
		}
		if ( i + 1 < row ) {
		    if ( j - 1 >= 0 ) {
			k0 += 5 * image[(i+1)*column + j-1];
			k4 += -3 * image[(i+1)*column + j-1];
		    }
		    if ( j + 1 < column) {
			k0 += -3 * image[(i+1)*column + j+1];
			k4 += 5 * image[(i+1)*column + j+1];
		    }
		    k0 += -3 * image[(i+1)*column + j];
		    k4 += -3 * image[(i+1)*column + j];
		}
		if ( j - 1 >= 0 ) {
		    k0 += 5 * image[i * column + j-1];
		    k4 += -3 * image[i * column + j-1];
		}
		if ( j + 1 < column) {
		    k0 += -3 * image[i * column + j+1];
		    k4 += 5 * image[i * column + j+1];
		}
		features[i*column +j] = max(k0, k4);
		//cout << "k0 = " << k0 << ", k4 = " << k4 << ", max = " << max(k0, k4) << endl;
	    }
	}
    }
    else if ( 2 == direction ) {
	for (int i = 0; i < row; ++i ) {
	    for ( int j = 0; j < column; ++j) {
		//cout << "i = " << i << ", j = " << j << " : ";
		int k0 = 0, k4 = 0;
		if (i - 1 >= 0) {
		    if ( j - 1 >= 0 ) {
			k0 += 5 * image[(i-1)*column + j-1];
			k4 += -3 * image[(i-1)*column + j-1];
		    }
		    if ( j + 1 < column ) {
			k0 += -3 * image[(i-1)*column + j+1];
			k4 += -3 * image[(i-1)*column + j+1];
		    }
		    k0 += 5 * image[(i-1)*column + j];
		    k4 += -3 * image[(i-1)*column + j];
		}
		if ( i + 1 < row ) {
		    if ( j - 1 >= 0 ) {
			k0 += -3 * image[(i+1)*column + j-1];
			k4 += -3 * image[(i+1)*column + j-1];
		    }
		    if ( j + 1 < column) {
			k0 += -3 * image[(i+1)*column + j+1];
			k4 += 5 * image[(i+1)*column + j+1];
		    }
		    k0 += -3 * image[(i+1)*column + j];
		    k4 += 5 * image[(i+1)*column + j];
		}
		if ( j - 1 >= 0 ) {
		    k0 += 5 * image[i * column + j-1];
		    k4 += -3 * image[i * column + j-1];
		}
		if ( j + 1 < column) {
		    k0 += -3 * image[i * column + j+1];
		    k4 += 5 * image[i * column + j+1];
		}
		features[i*column +j] = max(k0, k4);
		//cout << "k0 = " << k0 << ", k4 = " << k4 << ", max = " << max(k0, k4) << endl;
	    }
	}	
    }
    else if ( 3 == direction ) {
	for (int i = 0; i < row; ++i ) {
	    for ( int j = 0; j < column; ++j) {
		//cout << "i = " << i << ", j = " << j << " : ";
		int k0 = 0, k4 = 0;
		if (i - 1 >= 0) {
		    if ( j - 1 >= 0 ) {
			k0 += -3 * image[(i-1)*column + j-1];
			k4 += -3 * image[(i-1)*column + j-1];
		    }
		    if ( j + 1 < column ) {
			k0 += 5 * image[(i-1)*column + j+1];
			k4 += -3 * image[(i-1)*column + j+1];
		    }
		    k0 += 5 * image[(i-1)*column + j];
		    k4 += -3 * image[(i-1)*column + j];
		}
		if ( i + 1 < row ) {
		    if ( j - 1 >= 0 ) {
			k0 += -3 * image[(i+1)*column + j-1];
			k4 += 5 * image[(i+1)*column + j-1];
		    }
		    if ( j + 1 < column) {
			k0 += -3 * image[(i+1)*column + j+1];
			k4 += -3 * image[(i+1)*column + j+1];
		    }
		    k0 += -3 * image[(i+1)*column + j];
		    k4 += 5 * image[(i+1)*column + j];
		}
		if ( j - 1 >= 0 ) {
		    k0 += -3 * image[i * column + j-1];
		    k4 += 5 * image[i * column + j-1];
		}
		if ( j + 1 < column) {
		    k0 += 5 * image[i * column + j+1];
		    k4 += -3 * image[i * column + j+1];
		}
		features[i*column +j] = max(k0, k4);
		//cout << "k0 = " << k0 << ", k4 = " << k4 << ", max = " << max(k0, k4) << endl;
	    }
	}	
    }
    //  for (int i = 0; i < row*column; ++i) {
    //	if ( 0 == i % column)
    //	    cout << endl;
    //	cout << features[i] << " ";
    // }
    //  cout << "end " << endl;
}


int Recognizer::sixteen_into_one(int* old, int column,  int min_row_index, int min_column_index) const
{
    int max = old[0];
    for (int i = min_row_index; i < min_row_index + 4; ++i ) {
	for ( int j = min_column_index; j < min_column_index + 4; ++j) {
	    if ( max < old[ i * column + j] )
		max = old[i * column + j];
	}
    }
    return max;
}


void Recognizer::compress(int* old, int row, int column, double* now) const 
{
    if ( nullptr == now) {
	printf ("compress error 0\n");
	return;
    }
    if (0 != row % 4 || 0 != column % 4 ) {
	printf ("compress error 0 \n");
	return;
    }
    for (int i = 0; i < row / 4;  ++i) {
	for ( int j = 0; j < column / 4; ++j) {
	    double td = sixteen_into_one(old, column, i * 4, j * 4);
	    //cout << "max " << td << endl;
	    if ( td < Epsilon )
		now[i * 7 + j] = 0;
	    else
		now[i * 7 + j] = td / 6375;
	}
    }
}


void Recognizer::features_extraction(unsigned char* image, int row, int column, int direction, double* features) const
{
    if (nullptr == features) {
	printf ("features_extraction error 0\n");
	return;
    }
    int* temp = new int[column * row];
    if (nullptr == temp ) {
	printf ("features_extraction error 1\n");
	return ;
    }
    if ( direction < 4 ) {
	Kirsch(image, row, column, direction, temp);
	compress(temp, row, column, features);
    }
    else if ( 4 == direction ) {
	//
	canny(image, row, column, image);
	contour_descriptors(image, row, column, features);
	//my_descriptors(image, row, column, features);
    }
    delete temp;
}


int Recognizer::classification(unsigned char* image) const
{
    //cout << "classification : ";
    int type = 0;
    unordered_map<int, int> map1;
    double res[10];
    double input2[50];
    int count = 0;
    for (int i = 0; i < 5; ++i ) {
	double inputs[49];
	int num = 0;
	//Kirsch(image, 20, 20, i, inputs, num);
	features_extraction(image, 28, 28, i, inputs);
	fann_type *out = fann_run(ann[i], inputs);
	for ( int ii = 0; ii < 10; ++ii ) {
	    input2[count++] = out[ii];
	}
    }
    fann_type *out = fann_run(top_ann, input2);
    double max = 0;
    double second_max = 0;
    for ( int i = 0; i < 10; ++i )
    {
	if ( max < out[i] ) {
	    max = out[i];
	    type = i;
	}
	else if ( second_max < out[i] ) {
	    second_max = out[i];
	}
    }
    if ( max - second_max < threshold1) {
	return -1;
    }
    return type;
}


void Recognizer::load_nn()
{
    for ( int i = 0; i < NN_NUM; ++i ) {
	stringstream ss;
	ss << i;
	if ( nullptr != ann[i] ) {
	    fann_destroy(ann[i]);
	    ann[i] = nullptr;
	}
	string path = working_path+"nn"+ss.str()+".net";
	ann[i] = fann_create_from_file(path.c_str());
	if ( nullptr == ann[i]) {
	    printf ("load nn error\n");
	    return;
	}
    }
}


void Recognizer::load_nn_all()
{
    for ( int i = 0; i < NN_NUM; ++i ) {
	stringstream ss;
	ss << i;
	if ( nullptr != ann[i] ) {
	    fann_destroy(ann[i]);
	    ann[i] = nullptr;
	}
	string path = working_path+"nn"+ss.str()+".net";
	ann[i] = fann_create_from_file(path.c_str());
	if ( nullptr == ann[i]) {
	    printf ("load nn error\n");
	    return;
	}
    }
    top_ann = fann_create_from_file((working_path + "top.net").c_str());
    if ( nullptr == top_ann ) {
	printf ("load top_ann error\n");
	return;
    }
}


void Recognizer::test(const string& imagef, const string& labelf)
{
    using std::chrono::system_clock;
    unordered_map<int, int> correct;
    int error = 0;
    unordered_map<int, int> classification_map;
    unordered_map<int, int> class_map;
    ifstream iimagef(imagef);
    if ( iimagef.fail() ) {
	printf ("open file error %s\n", imagef.c_str());
	return;
    }
    ifstream ilabelf(labelf);
    if ( ilabelf.fail() ) {
	printf ("open file error %s\n", labelf.c_str());
	return;
    }

    char bit_32[4];
    //read MSB
    iimagef.read(bit_32, 4);
    ilabelf.read(bit_32, 4);
    //read number of items
    iimagef.read(bit_32, 4);
    ilabelf.read(bit_32, 4);
    const int size = 60000;
    //read nuber of rows (only in images)
    iimagef.read(bit_32, 4);
    const int rows = 28;
    //read number of columns (only in images)
    iimagef.read(bit_32, 4);
    const int columns = 28;
    const int inputs_d = rows * columns;
    cout << "bit_32 : " <<(int)bit_32[3] << endl;
    char byte[1];
    int count_classifiers = 0;
    int count = 0;
    int count_byte = 0;
    unsigned char image[784];

    using namespace chrono;
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    while ( !iimagef.eof() ) {
	for ( int i = 0; i < inputs_d; ++i ) {
	    //iimagef >> ubyte;
	    iimagef.read(byte, 1);
	    if ( iimagef.eof() )
		break;
	    ++count_byte;
	    unsigned char ubyte = (unsigned char)byte[0];
	    image[i] = ubyte;
	}
	++count;
	ilabelf.read(byte, 1);
	if (count < 5000)
	    continue;
	int type = classification(image);
	count_classifiers++;
	classification_map[type]++;
	class_map[(int)byte[0]]++;
	if ( type >= 0 && type != (int)byte[0] ) {
	    error++;
	}
	else if ( type >= 0 && type == (int)byte[0] ) {
	    correct[(int)byte[0]]++;
	}
    }
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
    // delete inputs;
    system_clock::time_point today = system_clock::now();
    std::time_t tt = system_clock::to_time_t (today);
    ofstream outfile(working_path+"result.txt", ios::app);
    outfile << string(ctime(&tt));
    //outfile << "ann\t inputs\t" << ann->num_input;
    //for (auto p = hiddens.begin(); p != hiddens.end(); ++p) {
    //	outfile << "\thidden 1: " << *p;
    //  }
    //outfile << "\toutputs : " << ann->num_output << endl; 

    outfile << "count " << count-1 << endl;
    outfile <<"it took \t" << time_span.count() << "s\tSpeed\t" << count_classifiers/(double)time_span.count() << "images/s" <<  endl;
    int total_correct = 0;
    int total_test = 0;
    for (auto p = class_map.begin(); p != class_map.end(); ++p) {
	total_test += p->second;
	total_correct += correct[p->first];
	outfile << p->first << " : ";
	double precision = correct[p->first] / (double)classification_map[p->first];
	double recall = correct[p->first] / (double)p->second;

	outfile << "precision " << precision;
	outfile << "\trecall " << recall;
	outfile << "\tF1 " << 2 * precision * recall / (precision + recall) << endl;
    }
    int total_classification = 0;
    for ( auto p = classification_map.begin(); p != classification_map.end(); ++p ) {
	if ( p->first >= 0)
	    total_classification += p->second;
    }
    cout << total_test << " " << count_classifiers << endl;
    outfile << "threshold\t" << threshold1 << endl;
    outfile << "total\t" << total_correct / (double)total_test << endl;
    outfile << "total error\t" << error / (double)total_test << endl;
    outfile << "corrects for classification\t" << total_correct / (double)total_classification << endl;
    //outfile << "total error\t" << error / (double)total_ << endl;
    outfile << "total reject\t" << classification_map[-1] / (double)total_test << endl << endl << endl;
}


void Recognizer::canny(unsigned char *image, int row, int column, unsigned char* new_image) const
{
    const int threshold = 90;
    using namespace cv;
    Mat img(Size(row, column), CV_8UC1, image);
    Canny(img, img, threshold, threshold * 3, 3);
    /*    int N = 0;
	  int nn = 0;
	  for (int i = 0; i < row; ++i ) {
	  int b = i % 2;
	  for (int j = 0; j < column; ++j ) {
	  int index = i * column + j + b;
	  if (index > column * row)
	  continue;
	  if (img.data[index] > 0)
	  nn++;
	  if ( 0 != j % 2 ) {
	  img.data[index] = 0;
	  }
	  new_image[index] = img.data[index];
	  if (img.data[index] > 0)
	  N++;
	  }
	  }*/
    //namedWindow("2", 2);
    //imshow("2", img);
    //cout << "N " << N << " " << nn <<  endl;
}


void Recognizer::contour_descriptors(unsigned char *image, int row, int column, double* features) const
{
    vector<pair<double, double>> c, C;
    //Fourier descriptors
    double amplitude = 0;
    for ( int i = 0; i < row; ++i ) {
	for ( int j = 0; j < column; ++j ) {
	    if ( image[i * column + j] > 0 ) {
		pair<double, double> ck;
		ck.first = pow(i * i + j * j, 0.5);
		amplitude += ck.first;
		ck.second = acos(i/ck.first);
		c.push_back(ck);
	    }
	}
    }
    int index = 0;
    double t = -2 * Pi / c.size();
    for (int u = 0; u < 16 && index < 48; ++u ) {
	pair<double, double> p;
	for ( int k = 0; k < c.size(); ++k ) {
	    //double theta =  u * k * t + c[k].second;
	    double theta = (c.size() - 1 - u) * k * t + c[k].second;
	    double r = c[k].first;
	    p.first += r * cos(theta);
	    p.second += r * sin(theta);
	}
	double r = pow((p.first * p.first + p.second *  p.second), 0.5);
	features[index++] = r / amplitude;
	features[index++] = acos( p.first / r ) / Pi;
    }

    double m00, m10, m01;
    CvMoments moment;
    Mat img(Size(28, 28), CV_8UC1, image);
    IplImage src = img;
    cvMoments(&src, &moment, 2);
    m00 = cvGetSpatialMoment(&moment, 0, 0);
    features[index++] = sigmod(cvGetSpatialMoment(&moment, 0, 0));
    features[index++] = sigmod(cvGetSpatialMoment(&moment, 1, 0));
    features[index++] = sigmod(cvGetSpatialMoment(&moment, 0, 1));
    features[index++] = sigmod(cvGetCentralMoment(&moment, 2, 0));
    features[index++] = sigmod(cvGetCentralMoment(&moment, 1, 1));
    features[index++] = sigmod(cvGetCentralMoment(&moment, 0, 2));
    features[index++] = sigmod(cvGetSpatialMoment(&moment, 3, 0));
    features[index++] = sigmod(cvGetCentralMoment(&moment, 2, 1));
    features[index++] = sigmod(cvGetCentralMoment(&moment, 1, 2));
    features[index++] = sigmod(cvGetCentralMoment(&moment, 0, 3));
    //cout << "m00 " << m00 << endl;
    CvHuMoments humoment;
    cvGetHuMoments(&moment, &humoment);
    //cout << "hu :" << humoment.hu1 << endl;
    features[index++] = sigmod(humoment.hu1);
    for (; index < 49; ++index) {
	features[index] = 0.0;
    }
}


void Recognizer::my_descriptors(unsigned char *image, int row, int column, double* features) const
{
    int index = 0;
    for (int i = 0; i < row; ++i ) {
	for ( int j = 0; j < column; ++j ) {
	    if ( image[i * column + j] > 0 && index < 48 ) {
		features[index++] = i / (double)28;
		features[index++] = j / (double)28;
	    }
	}
    }
    for (; index < 49; ++index)
	features[index] = 0;
}

