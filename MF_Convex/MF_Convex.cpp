#include "MF_Convex.h"

/* Debugging option */
// #define RECURSION_TRACE
// #define FIRST_SUBPROBLEM_DEBUG
// #define SECOND_SUBPROBLEM_DEBUG

void usage () { 
    cout << "./MSA_Convex (options) [seq_file]" << endl;
    cout << "seq_file should contain one or more DNA sequence. " << endl;
    cout << "Options: " << endl;
    cout << "\t-L Set the lambda, which is the weight for || W ||_p. (default 1.0)" << endl;
    cout << "\t-l Set the minimal length of atoms. (default 2)" << endl;
    cout << "\t-u Set the maximum length of atoms. (default 3)" << endl;
    cout << "\t-r Set step size (\\rho) for updating the ADMM coordinate variables. (default 0.1)"<< endl;
    cout << "\t-p Set maximum pertubation of penalty to break ties. (default 0)"<< endl;
    cout << "\t-s Set ADMM early stop toggle: early stop (on) if > 0. (default on)"<< endl;
    cout << "\t-r Set whether reinitialize W_1 and W_2 at each ADMM iteration. (default off)"<< endl;
}

void parse_cmd_line (int argn, char** argv) {
    if (argn < 2) { 
        usage();
        exit(0);
    }
    int i;
    for(i = 1; i < argn; i++){
        if ( argv[i][0] != '-' ) break;
        if ( ++i >= argn ) usage();
        switch(argv[i-1][1]){
            case 'e': ADMM_EARLY_STOP_TOGGLE = (atoi(argv[i])>0); break;
            case 'L': LAMBDA = atof(argv[i]); break;
            case 'l': L_MIN = atof(argv[i]); break;
            case 'u': L_MAX = atoi(argv[i]); break;
            case 'r': RHO = atof(argv[i]); break;
            case 'p': PERB_EPS = atof(argv[i]); break;
            case 'i': REINIT_W_ZERO_TOGGLE = (atoi(argv[i])>0); break;
            default:
                      cerr << "unknown option: -" << argv[i-1][1] << endl;
                      usage();
                      exit(0);
        }
    }
    if (i >= argn) usage();
    trainFname = argv[i];
}

bool double_dec_comp (double firstElem, double secondElem) {
	// sort pairs by second element with *decreasing order*
	return firstElem > secondElem;
}

void proximal (Tensor* wbar, double lambda, bool& all_zeros) {
    vector<double> alpha_vec;
    int num_alpha_elem = 0;
    int num_elem = 0;
    int T1 = wbar->size(); 
    // 1. first scan: check active entry
    for (int n = 0; n < T1; n ++) {
        int T2 = (*wbar)[n].size();
        for (int i = 0; i < T2; i ++) {
            int T3 = (*wbar)[n][i].size();
            num_elem += T3;
            for (int j = 0; j < T3; j ++) {
                if ((*wbar)[n][i][j] > 0.0) {
                    alpha_vec.push_back ((*wbar)[n][i][j]);
                    ++ num_alpha_elem;
                }
            }
        }
    }
    if (num_alpha_elem == 0) {
        all_zeros = true;
        return;
    }
    // 2. sorting
    std::sort (alpha_vec.begin(), alpha_vec.end(), double_dec_comp);
    // 3. find mstar
    double max_term = -1e50;
    double separator = 0.0;
    int mstar = 0; // number of elements support the sky
    double new_term, sum_alpha = 0.0;
    for (int i = 0; i < num_alpha_elem; i ++) {
        sum_alpha += alpha_vec[i];
        new_term = (sum_alpha - lambda) / (i + 1.0);
        if ( new_term > max_term ) {
            separator = alpha_vec[i];
            max_term = new_term;
            mstar = i;
        }
    }
    if (max_term < 0) max_term = (sum_alpha - lambda) / num_elem;
    if (max_term < 0) {
        all_zeros = true;
        return;
    }
    // 4. second scan: assign closed-form solution to wbar
    for (int n = 0; n < T1; n ++) {
        int T2 = (*wbar)[n].size();
        for (int i = 0; i < T2; i ++) {
            int T3 = (*wbar)[n][i].size();
            for (int j = 0; j < T3; j ++) {
                if ( abs((*wbar)[n][i][j]) >= separator ) 
                    (*wbar)[n][i][j] = max_term;
                else 
                    (*wbar)[n][i][j] = max((*wbar)[n][i][j], 0.0);
            }
        }
    }
}

/* Subproblem 2: update W_2 */
void suppress (TensorMap& W1, TensorMap& W2, TensorMap& Y, double rho, vector<int>& lenSeqs, double lambda) {
    int numSeqs = lenSeqs.size();
    vector<string> to_remove;
    for (auto it=Y.begin(); it!=Y.end(); it++) {
        string atom = it->first;
        Tensor* Yt  = it->second;
        Tensor* W1t = (W1.find(atom) != W1.end())? W1[atom] : NULL;
        Tensor* W2t = (W2.find(atom) != W2.end())? W2[atom] : NULL;
        // if W2 has no such atom, create one with all zeros
        if (W2t == NULL) {
            W2t = new Tensor();
            for (int n = 0; n < numSeqs; n ++) {
                Matrix tmp (lenSeqs[n], vector<double>(L_MAX, 0.0));
                W2t->push_back(tmp);
            }
            W2[atom] = W2t;
        }
        if (W1t == NULL) {
            // W2[atom] = - Y[atom] / rho
            tensor_scalar_mult (W2t, -1.0/rho, Yt);
        } else {
            // W2[atom] = W1[atom] - Y[atom] / rho
            tensor_ratio_add (W2t, W1t, -1.0/rho, Yt);
        }
        // proximal method to find close-form solution
        bool all_zeros = false;
        proximal (W2t, lambda, all_zeros);
        if (all_zeros) to_remove.push_back(atom);        
    }
    int num_to_remove = to_remove.size();
    for (int i = 0; i < num_to_remove; i ++) {
        string atom = to_remove[i];
        if (W2.find(atom) != W2.end()) free(W2[atom]);
        W2.erase(atom);
    }
}

void viterbi(TensorMap& W1, SequenceSet& allSeqs, vector<int>& lenSeqs) {
    // TODO:
    ;
}

/* Subproblem 1: update W_1 */
void align (TensorMap& W1, TensorMap& W2, TensorMap& Y, double rho, SequenceSet& allSeqs, vector<int>& lenSeqs) {
    // frank-wolfe
    int numSeqs = lenSeqs.size();
    int fw_iter = -1;
    while (fw_iter < MAX_1st_FW_ITER) {
        fw_iter ++;
        // 1. find alignment: brute-force search
        TensorMap S;
        viterbi(W1, allSeqs, lenSeqs);

        // 2. Exact Line search: determine the optimal step size \gamma
        double numerator = 0.0, denominator = 0.0;
        // TODO: 

        // Early Stop condition A: neglible denominator
        // if (denominator < 1e-6) break; // TODO
        double gamma = numerator / denominator;
        // initially pre-set to Conv(A)
        if (fw_iter == 0) gamma = 1.0;
        // Gamma should be bounded by [0,1]
        gamma = max(gamma, 0.0);
        gamma = min(gamma, 1.0);
        // Early Stop condition B: neglible gamma
        // if (fabs(gamma) < EPS_1st_FW) break;  // TODO

        // 3. update W1: W1 = (1-gamma) * W1 + gamma * S
        double one_minus_gamma = 1-gamma;
        for (auto it=W1.begin(); it !=W1.end(); it++) {
            string atom = it->first;
            Tensor* W1t = W1[atom];
            // W1 += (1-gamma) * W1
            tensor_axpy(W1t, one_minus_gamma, W1t);
        }
        for (auto it=S.begin(); it !=S.end(); it++) {
            string atom = it->first;
            Tensor* W1t = W1[atom];
            Tensor* St  = S [atom];
            if (W1t == NULL) {
                W1t = new Tensor();
                for (int n = 0; n < numSeqs; n ++) {
                    Matrix tmp (lenSeqs[n], vector<double>(L_MAX, 0.0));
                    W1t->push_back(tmp);
                }
                W1[atom] = W1t;
            }
            // W1 += gamma * S
            tensor_axpy(W1t, gamma, St);
        }

    }
}

void coordinate (TensorMap& Y, TensorMap& W1, TensorMap& W2, double rho, vector<int>& lenSeqs) {
    int numSeqs = lenSeqs.size();
    for (auto it=W1.begin(); it !=W1.end(); it++) {
        string atom = it->first;
        Tensor* W1t = W1[atom];
        Tensor* W2t = ( W2.find(atom) != W2.end() ) ? W2[atom] : NULL;
        Tensor* Yt  = (  Y.find(atom) !=  Y.end() ) ?  Y[atom] : NULL;
        if (Yt == NULL) { 
            Yt = new Tensor();
            for (int n = 0; n < numSeqs; n++) {
                Matrix tmp (lenSeqs[n], vector<double>(L_MAX, 0.0));
                Yt->push_back(tmp);
            }
            Y[atom] = Yt;
        } 
        if (W2t == NULL) {
            // axpy: Y += rho * W_1
            tensor_axpy(Yt, rho, W1t);
        } else {
            // axpy: Y += rho * (W_1 - W_2)
            tensor_diff_axpy(Yt, rho, W1t, W2t);
        }
    }
}

TensorMap CVX_ADMM_MF (SequenceSet& allSeqs, vector<int>& lenSeqs) {
    /*{{{*/
    // 1. initialization
    int numSeqs = allSeqs.size();
    TensorMap W_1, W_2, Y; 

    // 2. ADMM iteration
    int iter = 0;
    double rho = RHO, lambda = LAMBDA;
    double prev_CoZ = MAX_DOUBLE;
    while (iter < MAX_ADMM_ITER) {

        // 2a. Subprogram: FrankWolf Algorithm, row separable
        vector<MatrixMap> sub_W_1 (numSeqs); 
        for (int n = 0; n < numSeqs; n ++) {
            align (W_1, W_2, Y, rho, allSeqs, lenSeqs);
        }
        // combine();

        // 2b. Subprogram: proximal method, column separable
        suppress (W_1, W_2, Y, rho, lenSeqs, lambda);

        // 2d. update Y: Y += rho * (W_1 - W_2)
        coordinate (Y, W_1, W_2, rho, lenSeqs);

        // 2e. print out tracking info
        /*
        double CoZ = 0.0;
        for (int n = 0; n < numSeq; n++) 
            CoZ += tensor4D_frob_prod(C[n], W_2[n]);

        char COZ_val [50], w1mw2_val [50]; 
        sprintf(COZ_val, "%6f", CoZ);
        sprintf(w1mw2_val, "%6f", W1mW2);
        cerr << "ADMM_iter = " << iter 
            << ", C o Z = " << COZ_val
            << ", Wdiff_max = " << w1mw2_val
            << ", obj_rounded = " << obj_rounded
            << endl;
            */

        // 2f. stopping conditions
        /*
        if (ADMM_EARLY_STOP_TOGGLE and iter > MIN_ADMM_ITER)
            if ( W1mW2 < EPS_Wdiff ) {
                cerr << "CoZ Converges. ADMM early stop!" << endl;
                break;
            }
        prev_CoZ = CoZ;
        */
        iter ++;
    }
    /*
    cout << "W_1: " << endl;
    for (int i = 0; i < numSeq; i ++) tensor4D_dump(W_1[i]);
    cout << "W_2: " << endl;
    for (int i = 0; i < numSeq; i ++) tensor4D_dump(W_2[i]);
    return W_2;
    */
    /*}}}*/
}


int main (int argn, char** argv) {
    // 1. parse cmd 
    parse_cmd_line(argn, argv);
    // 2. input DNA sequence file
    int numSeq = 0;
    SequenceSet allSeqs (0, Sequence());
    parse_seqs_file(allSeqs, numSeq, trainFname);
    vector<int> lenSeqs (numSeq, 0);
    for (int n = 0; n < numSeq; n ++) 
        lenSeqs[n] = allSeqs[n].size();

    // pre-info
    cout << "#########################################################" << endl;
    cout << "ScoreMatch: " << C_M;
    cout << ", ScoreMismatch: " << C_MM << endl;
    /*
    cout << "PERB_EPS: " << PERB_EPS;
    cout << ", FW1_GFW_EPS: " << FW1_GFW_EPS;
    cout << ", FW2_GFW_EPS: " << FW2_GFW_EPS;
    cout << ", LENGTH_OFFSET: " << LENGTH_OFFSET;
    cout << ", EPS_Wdiff: " << EPS_Wdiff << endl;
    */
    for (int n = 0; n < numSeq; n ++) sequence_dump(allSeqs, n);
    cout << "#########################################################" << endl;

    // 3. relaxed convex program: ADMM-based algorithm
    // omp_set_num_threads(NUM_THREADS);
    time_t begin = time(NULL);
    TensorMap W = CVX_ADMM_MF (allSeqs, lenSeqs);
    time_t end = time(NULL);

    cout << "Time Spent: " << end - begin << " seconds" << endl;
    return 0;
}
