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
#include <sstream> 
#include <cmath>

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
const int MAX_2nd_FW_ITER = 300;
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
/*{{{*/
class Cell {
    public:
        double score;   
        Action action;
        int dim;   
        vector<int> location; 
        char acidA, acidB;
        int ans_idx;
        Cell (int dim) {
            this->score = 0;
            this->action = UNDEFINED; this->dim = dim; 
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
            s << action2str(this->action) << ", ";
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

typedef vector<vector<double> > Matrix; // 2-d double matrix
typedef vector<Matrix> Tensor;  // 3-d double tensor
typedef map<string, *Tensor> Tensor4D; // 4-d double Tensor
/*}}}*/

// C is the tensor specifying the penalties 
void set_C (Tensor5D& C, SequenceSet& allSeqs) {
/*{{{*/
    int T0 = C.size();
    int T2 = C[0][0].size();
    int T3 = NUM_DNA_TYPE;
    int T4 = NUM_MOVEMENT;
    for (int n = 0; n < T0; n ++) {
        int T1 = C[n].size();
        for (int i = 0; i < T1; i ++) {
            for (int j = 0; j < T2; j ++) {
                for (int k = 0; k < T3; k ++) {
                    for (int m = 0; m < T4; m ++) {
                        if (m == INS_BASE_IDX) {
                            if (allSeqs[n][i] == '#') {
                                C[n][i][j][k][m] = HIGH_COST;
                                continue;
                            }
                            C[n][i][j][k][m] = C_I;
                        }
                        // DELETION PENALTIES
                        else if (m == DELETION_START) // disallow delete *
                            C[n][i][j][k][m] = HIGH_COST;
                        else if (m == DELETION_END) // disallow delete #
                            C[n][i][j][k][m] = HIGH_COST;
                        else if (DEL_BASE_IDX <= m and m < MTH_BASE_IDX) {
                            C[n][i][j][k][m] = C_D;
                        }
                        // MATCH PENALTIES
                        else if (m == MATCH_START)
                            C[n][i][j][k][m] = (allSeqs[n][i] == '*')?NO_COST:HIGH_COST; // disallow mismatch *
                        else if (m == MATCH_END) 
                            C[n][i][j][k][m] = (allSeqs[n][i] == '#')?NO_COST:HIGH_COST; // disallow mismatch #
                        else if (MTH_BASE_IDX <= m) {
                            if (allSeqs[n][i] == '#' and m != MATCH_END) {
                                C[n][i][j][k][m] = HIGH_COST;
                                continue;
                            }
                            C[n][i][j][k][m] = (dna2T3idx(allSeqs[n][i]) == m-MTH_BASE_IDX)?C_M:C_MM;
                        }
                        C[n][i][j][k][m] += PERB_EPS * ((double) rand())/RAND_MAX;
                    }
                }
            }
        }
    }
/*}}}*/
}

/* Tensors auxiliary function */
/*{{{*/
void tensor4D_average (Tensor4D& dest, Tensor4D& src1, Tensor4D& src2) {
    int T1 = src1.size();
    for (int i = 0; i < T1; i ++) {
        int T2 = src1[i].size();
        for (int j = 0; j < T2; j ++) 
            for (int d = 0; d < NUM_DNA_TYPE; d ++) 
                for (int m = 0; m < NUM_MOVEMENT; m ++)
                    dest[i][j][d][m] = 0.5*(src1[i][j][d][m] + src2[i][j][d][m]);
    }
}
double tensor4D_frob_prod (Tensor4D& src1, Tensor4D& src2) {
    double prod = 0.0;
    int T1 = src1.size();
    for (int i = 0; i < T1; i ++) {
        int T2 = src1[i].size();
        for (int j = 0; j < T2; j ++) 
            for (int d = 0; d < NUM_DNA_TYPE; d ++) 
                for (int m = 0; m < NUM_MOVEMENT; m ++)
                    prod += src1[i][j][d][m] * src2[i][j][d][m];
    }
    return prod;
}
void tensor4D_lin_update (Tensor4D& dest, Tensor4D& src1, Tensor4D& src2, double ratio) {
    int T1 = src1.size();
    for (int i = 0; i < T1; i ++) {
        int T2 = src1[i].size();
        for (int j = 0; j < T2; j ++) 
            for (int d = 0; d < NUM_DNA_TYPE; d ++) 
                for (int m = 0; m < NUM_MOVEMENT; m ++)
                    dest[i][j][d][m] += ratio * (src1[i][j][d][m] - src2[i][j][d][m]);
    }
}
void tensor4D_copy (Tensor4D& dest, Tensor4D& src1) {
    int N = src1.size();
    int T1 = src1.size();
    for (int i = 0; i < T1; i ++) {
        int T2 = src1[i].size();
        for (int j = 0; j < T2; j ++) 
            for (int d = 0; d < NUM_DNA_TYPE; d ++) 
                for (int m = 0; m < NUM_MOVEMENT; m ++)
                    dest[i][j][d][m] = src1[i][j][d][m];
    }
    return ;
}
void tensor_dump (Tensor& W) {
    int T1 = W.size();
    for (int i = 0; i < T1; i ++) {
        int T2 = W[i].size();
        for (int j = 0; j < T2; j ++) 
            int T3 = W[i][0].size();
            for (int d = 0; d < T3; d ++) 
                    if (W[i][j][d] > 0.0)
                        cout << "(i="  << i
                            << ", j=" << j
                            << ", d=" << d
                            << ", m=" << m
                            << "): " << W[i][j][d][m]
                            << endl;
    }
}
/*}}}*/

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
void tensor_axpy (Tensor* sink, double ratio, Tensor* s1, Tensor* s2) {
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
