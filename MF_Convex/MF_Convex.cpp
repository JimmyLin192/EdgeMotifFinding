#include "MSA_Convex.h"

/* Debugging option */
// #define RECURSION_TRACE
// #define FIRST_SUBPROBLEM_DEBUG
// #define SECOND_SUBPROBLEM_DEBUG

void usage () { 
    cout << "./MSA_Convex (options) [seq_file]" << endl;
    cout << "seq_file should contain one or more DNA sequence. " << endl;
    cout << "Options: " << endl;
    cout << "\t-l Set the minimal length of atoms. (default 2)" << endl;
    cout << "\t-u Set the maximum length of atoms. (default 3)" << endl;
    cout << "\t-m Set step size (\\mu) for updating the ADMM coordinate variables. (default 0.1)"<< endl;
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
            case 'r': REINIT_W_ZERO_TOGGLE = (atoi(argv[i])>0); break;
            case 'l': L_MIN = atoi(argv[i]); break;
            case 'u': L_MAX = atoi(argv[i]); break;
            case 'm': MU = atof(argv[i]); break;
            case 'p': PERB_EPS = atof(argv[i]); break;
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

void proximal (Tensor* wbar, double lambda) {
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
    if (num_alpha_elem == 0) return;
    // 2. sorting
    std::sort (alpha_vec[j].begin(), alpha_vec[j].end(), double_dec_comp);
    // 3. find mstar
    double max_term = -1e50;
    double separator = 0.0;
    int mstar = 0; // number of elements support the sky
    double new_term, sum_alpha = 0.0;
    for (int i = 0; i < num_alpha_elem; i ++) {
        sum_alpha += alpha_vec[i];
        new_term = (sum_alpha - lambda) / (i + 1.0);
        if ( new_term > max_term ) {
            separator[j] = alpha_vec[i];
            max_term[j] = new_term;
            mstar = i;
        }
    }
    if (max_term[j] < 0) max_term = (sum_alpha - lambda) / num_elem;
    // 4. second scan: assign closed-form solution to wbar
    for (int n = 0; n < T1; n ++) {
        int T2 = (*wbar)[n].size();
        for (int i = 0; i < T2; i ++) {
            int T3 = (*wbar)[n][i].size();
            for (int j = 0; j < T3; j ++) {
                if ( max_term < 0 ) (*wbar)[n][i][j] = 0.0;
                if ( abs((*wbar)[n][i][j]) >= separator ) 
                    (*wbar)[n][i][j] = max_term;
                else 
                    (*wbar)[n][i][j] = max((*wbar)[n][i][j], 0.0);
            }
        }
    }
}

void reach_agreement (Tensor4D& W1, Tensor4D& W2, Tensor4D& Y, double rho, SequenceSet& allSeqs, vector<int>& lenSeqs) {
    int numAtoms = W1.size();
    for (auto it=Y.begin(); it!=Y.end(); it++) {
        string atom = it->first;
        Tensor* Yt  = it->second;
        Tensor* W1t = (W1.find(atom) != NULL)? W1[atom] : NULL;
        Tensor* W2t = (W2.find(atom) != NULL)? W2[atom] : NULL;
        // if W2 has no such atom, create one with all zeros
        if (W2t == NULL) {
            W2t = new Tensor(lenSeqs.size(), NULL);
            for (int seq_len : lenSeqs) {
                Matrix tmp (seq_len, vector<double>(L_MAX, 0.0));
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
        proximal (W2t);
    }
}

Tensor4D CVX_ADMM_MF (SequenceSet& allSeqs, vector<int>& lenSeqs) {
    /*{{{*/
    // 1. initialization
    int numSeq = allSeqs.size();
    Tensor4D C, W_1, W_2, Y; 
    // (0, Tensor(numSeq, Matrix(NUM_DNA_TYPE, vector<double>(L_MAX, 0.0))));  
   // set_C (C, allSeqs);

    // 2. ADMM iteration
    int iter = 0;
    double rho = RHO;
    double prev_CoZ = MAX_DOUBLE;
    while (iter < MAX_ADMM_ITER) {
        // 2a. Subprogram: FrankWolf Algorithm
        // NOTE: parallelize this for to enable parallelism
#ifdef PARRALLEL_COMPUTING
#pragma omp parallel for
#endif
        reach_alignment (W_1[n], W_2[n], Y[n], C[n], rho, allSeqs[n]);

        // 2b. Subprogram: 
        reach_agreement (W_1, W_2, Y, rho, allSeqs, lenSeqs);

        // 2d. update Y: Y += mu * (W_1 - W_2)
        // TODO: identity check
        for (int n = 0; n < numSeq; n ++)
            tensor4D_lin_update (Y[n], W_1[n], W_2[n], mu);

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
        if (ADMM_EARLY_STOP_TOGGLE and iter > MIN_ADMM_ITER)
            if ( W1mW2 < EPS_Wdiff ) {
                cerr << "CoZ Converges. ADMM early stop!" << endl;
                break;
            }
        prev_CoZ = CoZ;
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
    Tensor4D W = CVX_ADMM_MF (allSeqs, lenSeqs);
    time_t end = time(NULL);

    cout << "Time Spent: " << end - begin << " seconds" << endl;
    return 0;
}
