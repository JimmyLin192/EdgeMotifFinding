/* Imported Libraries */
using namespace std;
#include "stdio.h"
#include "stdlib.h"
#include <iostream>
#include <fstream>
#include <limits>
#include <vector>
#include <unordered_map>
#include <set>
#include <map>
#include <sstream> 
#include <cmath>
#include <cassert>

//#define CUBE_SMITH_WATERMAN_DEBUG
//#define PARRALLEL_COMPUTING

// variable parameters
char* trainFname = NULL;
int LENGTH_OFFSET = 0;
double RHO = 1.0;
double PERB_EPS = 0.0;
bool ADMM_EARLY_STOP_TOGGLE = true;
bool REINIT_W_ZERO_TOGGLE = true;

double LAMBDA = 1.0;
int L_MIN = 2;
int L_MAX = 3;

const int NUM_THREADS = 6;

/* Self-defined Constants and Global Variables */
const double MIN_DOUBLE = -1*1e99;
const double MAX_DOUBLE = 1e99; 
const double MAX_INT = numeric_limits<int>::max();

/* Algorithmic Setting */
const int MAX_1st_FW_ITER = 200;
const int MIN_ADMM_ITER = 10;
const int MAX_ADMM_ITER = 10000;
const double FW1_GFW_EPS = 1e-6;
const double FW2_GFW_EPS = 1e-3;
//const double EPS_ADMM_CoZ = 1e-5; 
const double EPS_Wdiff = 0.001;

/* Define Scores and Other Constants */
const char GAP_NOTATION = '-';
double C_MM = 0.0; // penalty of mismatch
double C_M = -1.0;    // penalty of match
const double HIGH_COST = 999999.0;
const double NO_COST = 0;

/* Data Structure */
class Cell {
    public:
        double score;   
        int dim;   
        vector<int> location; 
        char acidA, acidB;
        int ans_idx;
        Cell (int dim) {
            this->score = 0;
            this->dim = dim; 
            for (int i = 0; i < dim; i ++) location.push_back(-1);
            this->acidA = '?';
            this->acidB = '?';
            this->ans_idx = -1;
        }
        // convert to string:
        //    [(location vector), action, acidA, acidB, score] 
        string toString () {
            stringstream s;
            s << "[(";
            for (int i = 0 ; i < this->dim; i ++) {
                s << (this->location)[i];
                if (i < this->dim - 1) s << ",";
            }
            s << "), ";
            s << this->acidA << ", " << this->acidB;
            s << ", " << this->score << ") ";
            return s.str();
        }
};

typedef vector<vector<char> > SequenceSet;
typedef vector<char> Sequence;

typedef vector<Cell> Trace;  // for viterbi
typedef vector<Trace > Plane; // 2-d Cell Plane, for viterbi
typedef vector<Plane > Cube;  // 3-d Cell Cube

typedef vector<vector<double>> Matrix; // 2-d double matrix
typedef vector<Matrix> Tensor;  // 3-d double tensor
typedef map<string, Matrix*> MatrixMap; // 3-d double Tensor
typedef map<string, Tensor*> TensorMap; // 4-d double Tensor

/* Tensors auxiliary function */
bool all_zeros (Matrix& source) {
    int dim_x = source.size();
    for (int i = 0; i < dim_x; i ++) {
        int dim_y = source[i].size();
        for (int j = 0; j < dim_y; j ++) {
            if (abs(source[i][j]) > 1e-5) return false;
        }
    }
    return true;
}
void mat_copy (Matrix& sink, Matrix& source) {
    assert (source.size() == sink.size());
    for (int i = 0; i < source.size(); i ++) {
        assert (source[i].size() == sink[i].size());
        for (int j = 0; j < source[i].size(); j ++) {
            sink[i][j] = source[i][j];
        }
    }
}
void TensorMap2MatrixMap(int nid, MatrixMap& mmap, TensorMap& tmap) {
    for (auto it=tmap.begin(); it!=tmap.end(); it++) {
        string atom = it->first;
        Tensor* tensor = it->second;
        if (!all_zeros((*tensor)[n])) {
            mmap[atom] = &((*tensor)[n]);
        }
    }
}
void MatrixMap2TensorMap(TensorMap& tmap, vector<MatrixMap>& mmaps, vector<int>& lenSeqs) {
    // clear zero for tmap
    
    // copy info from mmaps to tmap
    for (int n = 0; n < mmaps.size(); n ++) {
        for (auto it=mmap[n].begin(); it!=mmap[n].end(); it++) {
            string atom = it->first;
            Matrix* mat = it->second;
            if (tmap.find(atom) == W1.end()) {
                Tensor* tmap_t = new Tensor();
                for (int i = 0; i < numSeqs; i ++) {
                    Matrix tmp (lenSeqs[i], vector<double>(L_MAX, 0.0));
                    tmap_t->push_back(tmp);
                }
                tmap[atom] = tmap_t;
            }
            mat_copy ((*(tmap[atom]))[n], *mat);
        }
    }
}
void dump (Matrix& source, string name) {
    int dim_x = source.size();
    cout << "===> dumping Matrix " <<  name  << ":" << endl;
    for (int i = 0; i < dim_x; i ++) {
        int dim_y = source[i].size();
        for (int j = 0; j < dim_y; j ++) 
            cout << " " << source[i][j];
        cout << endl;
    }
}
void dump (Tensor& source, string name) {
    int dim_x = source.size();
    cout << "===> dumping Tensor " <<  name  << ":" << endl;
    for (int i = 0; i < dim_x; i ++) {
        cout << "n = " << i << endl;
        int dim_y = source[i].size();
        for (int j = 0; j < dim_y; j ++) {
            int dim_z = source[i][j].size();
            for (int k = 0; k < dim_z; k ++) 
                cout << " " << source[i][j][k];
            cout << endl;
        }
    }
}
void dump (MatrixMap& source, string name) {
    cout << "=====> dumping MatrixMap " <<  name  << ":" << endl;
    for (auto it = source.begin(); it != source.end(); it ++) {
        dump(*(it->second), it->first);
    }
}
void dump (TensorMap& source, string name) {
    cout << "=====> dumping TensorMap " <<  name  << ":" << endl;
    for (auto it = source.begin(); it != source.end(); it ++) {
        dump(*(it->second), it->first);
    }
}
void sequence_dump (SequenceSet& allSeqs, int n) {
    char buffer [50];
    sprintf (buffer, "Seq%5d", n);
    cout << buffer << ": ";
    for (int j = 0; j < allSeqs[n].size(); j ++) 
        cout << allSeqs[n][j];
    cout << endl;
}
void parse_seqs_file (SequenceSet& allSeqs, int& numSeq, char* fname) {
    ifstream seq_file(fname);
    string tmp_str;
    while (getline(seq_file, tmp_str)) {
        int seq_len = tmp_str.size();
        Sequence ht_tmp_seq (seq_len+1+1, 0);
        ht_tmp_seq[0] = '*';
        for(int i = 0; i < seq_len; i ++) 
            ht_tmp_seq[i+1] = tmp_str.at(i);
        ht_tmp_seq[seq_len+1] = '#';
        allSeqs.push_back(ht_tmp_seq);
        ++ numSeq;
    }
    seq_file.close();
}

// sink += mu * source
void tensor_axpy (Tensor* sink, double ratio, Tensor* source) {
    for (int i = 0; i < source->size(); i ++) { // num_seqs
        assert (sink->size() == source->size() && "sink == source->size() fails.");
        for (int j = 0; j < (*source)[i].size(); j ++) { // num_characters
            assert ((*source)[i].size() == (*sink)[i].size() && 
                    "(*source)[i].size() == (*sink)[i].size() fails.");
            for (int k = 0; k < (*source)[i][j].size(); k ++) {
                assert ((*source)[i][j].size() == (*sink)[i][j].size() && 
                        "(*source)[i][j].size() == (*sink)[i][j].size() fails.");
                (*sink)[i][j][k] += ratio * (*source)[i][j][k];
            }
        }
    }
}
// sink += mu * (s1 - s2)
void tensor_diff_axpy (Tensor* sink, double ratio, Tensor* s1, Tensor* s2) {
    for (int i = 0; i < s1->size(); i ++) { // num_seqs
        assert (sink->size() == s1->size() && "sink == s1->size() fails.");
        assert (sink->size() == s2->size() && "sink == s2->size() fails.");
        for (int j = 0; j < (*s1)[i].size(); j ++) { // num_characters
            assert ((*s1)[i].size() == (*sink)[i].size() && 
                    "(*s1)[i].size() == (*sink)[i].size() fails.");
            assert ((*s2)[i].size() == (*sink)[i].size() && 
                    "(*s2)[i].size() == (*sink)[i].size() fails.");
            for (int k = 0; k < (*s1)[i][j].size(); k ++) {
                assert ((*s1)[i][j].size() == (*sink)[i][j].size() && 
                        "(*s1)[i][j].size() == (*sink)[i][j].size() fails.");
                assert ((*s2)[i][j].size() == (*sink)[i][j].size() && 
                        "(*s2)[i][j].size() == (*sink)[i][j].size() fails.");
                (*sink)[i][j][k] += ratio * ( (*s1)[i][j][k] - (*s2)[i][j][k]);
            }
        }
    }
}

void tensor_scalar_mult (Tensor* sink, double ratio, Tensor* source) {
    for (int i = 0; i < source->size(); i ++) { // num_seqs
        assert (sink->size() == source->size() && "sink == source->size() fails.");
        for (int j = 0; j < (*source)[i].size(); j ++) { // num_characters
            assert ((*source)[i].size() == (*sink)[i].size() && 
                    "(*source)[i].size() == (*sink)[i].size() fails.");
            for (int k = 0; k < (*source)[i][j].size(); k ++) {
                assert ((*source)[i][j].size() == (*sink)[i][j].size() && 
                        "(*source)[i][j].size() == (*sink)[i][j].size() fails.");
                (*sink)[i][j][k] = ratio * (*source)[i][j][k];
            }
        }
    }
}
void tensor_ratio_add (Tensor* sink, Tensor* media, double ratio, Tensor* source) {
    for (int i = 0; i < source->size(); i ++) { // num_seqs
        assert (sink->size() == source->size() && "sink == source->size() fails.");
        assert (media->size() == source->size() && "media == source->size() fails.");
        for (int j = 0; j < (*source)[i].size(); j ++) { // num_characters
            assert ((*source)[i].size() == (*sink)[i].size() && 
                    "(*source)[i].size() == (*sink)[i].size() fails.");
            assert ((*source)[i].size() == (*media)[i].size() && 
                    "(*source)[i].size() == (*media)[i].size() fails.");
            for (int k = 0; k < (*source)[i][j].size(); k ++) {
                assert ((*source)[i][j].size() == (*sink)[i][j].size() && 
                        "(*source)[i][j].size() == (*sink)[i][j].size() fails.");
                assert ((*source)[i][j].size() == (*media)[i][j].size() && 
                        "(*source)[i][j].size() == (*media)[i][j].size() fails.");
                (*sink)[i][j][k] = (*media)[i][j][k] + ratio * (*source)[i][j][k];
            }
        }
    }
}
