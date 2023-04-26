#ifndef NORMALS_EST_HEADER
#define NORMALS_EST_HEADER


#include <Eigen/Dense>
#include <nanoflann.hpp>
#include <vector>
#include <fstream>
#include <omp.h>
#include <iostream>
#include <time.h>
#include <stdlib.h>
#include <string>
#include <unistd.h>

typedef Eigen::Vector3d Vector3;
typedef Eigen::Vector2d Vector2;
typedef Eigen::Matrix3d Matrix3;
typedef Eigen::Matrix2d Matrix2;
typedef Eigen::MatrixX3d MatrixX3;
typedef Eigen::MatrixX3i MatrixX3i;
typedef Eigen::VectorXd VectorX;
typedef Eigen::MatrixXd MatrixX;
typedef Eigen::Vector3i Vector3i;
typedef typename nanoflann::KDTreeEigenMatrixAdaptor< MatrixX3 > kd_tree;

#define DEG2RAD M_PI/180
#define RAD2DEG 180/M_PI

class HoughAccum{
	public:
		VectorX accum;
		MatrixX3 accum_vec;
		Matrix3 P;
		int A;
};

class NormEst{

	/* ----- class member functions ----- */
	public:
		/** \brief Set Ks
		 *
		 * \input[in] points
		 * \input[in] Ks Value
		 *
		 */
		void 
		setKs(int* array, int m);

		/** \brief Compute the size of Ks 
		 *
		 * \input[in] none
		 *
		 * \return size of Ks
		 */
		int 
		getKsSize() {return Ks.size();};

		/** \brief get Ks
		 *
		 * \input[in] points
		 * \input[in] Ks Value
		 *
		 */
		void 
		getKs(int* array, int m);

		/** \brief get density sensitive 
		 *
		 * \input[in] none
		 *
		 * \return true if density sensitive
		 */
		bool 
		getDensitySensitive() {return use_aniso;};

		/** \brief set density sensitive 
		 *
		 * \input[in] density sensitive
		 *
		 */
		void 
		setDensitySensitive(bool d_s) {use_aniso=d_s;};

		/** \brief Compute the size of point cloud
		 *
		 * \input[in] none
		 *
		 * \return size of point cloud
		 */
		int 
		getPCSize() {return _pc.rows();};

		/** \brief Compute the size of point cloud 
		 *
		 * \input[in] none
		 *
		 * \return size of point cloud
		 */
		int 
		getPCNormalsSize() {return _normals.rows();};

		/** \brief initialize
		 *
		 * \input[in] none
		 *
		 */
		void 
		initialize();

		/** \brief get batch
		 *
		 * \input[in] batch id
		 * \input[in] batch size
		 * \input[in] points
		 *
		 */
		void 
		getBatch(int batch_id, int batch_size, double* array);

		/** \brief set batch
		 *
		 * \input[in] batch id
		 * \input[in] batch size
		 * \input[in] points 
		 *
		 */
		void 
		setBatch(int batch_id, int batch_size, double* array);

		/** \brief set T
		 *
		 * \input[in] input T
		 *
		 */
		void 
		setT(int T_) {T=T_;};

		/** \brief set Kaniso
		 *
		 * \input[in] Kaniso
		 *
		 */
		void 
		setKaniso(int Kaniso) {K_aniso=Kaniso;};

		/** \brief set A
		 *
		 * \input[in] input A
		 *
		 */
		void 
		setA(int A_){A = A_;}

		/** \brief get T
		 *
		 * \input[in] none
		 *
		 * \return T
		 */
		const int 
		getT()const {return T;}

        /** \brief get Kaniso 
		 *
		 * \input[in] none
		 *
		 * \return Kaniso
		 */
		const int 
		getKaniso() const {return K_aniso;}

		/** \brief get A 
		 *
		 * \input[in] none
		 *
		 * \return A
		 */
		const int 
		getA() const{return A;}

		/** \brief get points
		 *
		 * \input[in] points
		 * \input[in] row
		 * \input[in] column
		 *
		 */
		void 
		getPoints(double* array, int m, int n);

		/** \brief get normals
		 *
		 * \input[in] points
		 * \input[in] row
		 * \input[in] column
		 *
		 */
		void 
		getNormals(double* array, int m, int n);

		/** \brief set points
		 *
		 * \input[in] points
		 * \input[in] row
		 * \input[in] column
		 *
		 */
		void 
		setPoints(double* array, int m, int n);

		/** \brief set normals
		 *
		 * \input[in] points
		 * \input[in] row
		 * \input[in] column
		 *
		 */
		void 
		setNormals(double* array, int m, int n);

		/** \brief point cloud matrix
		 *
		 * \input[in] none
		 *
		 * \return matrix of point cloud
		 */
		const MatrixX3& 
		pc() const{return _pc;}

        /** \brief normal matrix
		 *
		 * \input[in] none
		 *
		 * \return normals
		 */
		MatrixX3& 
		normals(){return _normals;}

		/** \brief generate train data
		 *
		 * \input[in] noise level
		 * \input[in] number points
		 * \input[in] points
		 * \input[in] batch size
		 *
		 * \return training data
		 */
		int 
		generateTrainAccRandomCorner(int noise_val, int n_points, double* array, double* array_gt);
		
		/** \brief load .xyz file
		 *
		 * \input[in] file name
		 *
		 */
		void 
		loadXYZ(const std::string& filename);

		
		/** \brief save .xyz file
		 *
		 * \input[in] file name
		 *
		 */
		void 
		saveXYZ(const std::string& filename);

		




	/* ----- class member variables ----- */
	private:
		/** \brief reference to the point cloud */
		MatrixX3 _pc;

		/** \brief reference to the normal cloud */
		MatrixX3 _normals;

		/** \brief number of triplets to pick */
		int T=1000;

		int K_aniso=5;

		/** \brief side_size of accumulator */
		int A=33;

		std::vector<float> proba_vector;
		std::vector<int> counts_generated_elems;

	public:

		std::vector<int> Ks; // TODO

		bool use_aniso; // TODO

		int maxK;
		kd_tree* tree;
		bool is_tree_initialized=false;
		unsigned int randPos;
		std::vector<HoughAccum> accums;
		std::vector<unsigned int> rand_ints;
};

class EstimationTools{
	/* ----- class member functions ----- */
	public:
		void fillAccum(HoughAccum& hd, std::vector<long int>& nbh, int nbh_size,
					const NormEst* est, unsigned int& randPos,
					const std::vector<float>& proba_vector,
					bool use_aniso,
					bool compute_P = true, Matrix3 P_ref=Matrix3());


		double searchKNN(const kd_tree& tree, const Vector3& pt, int K, std::vector<long int>& indices, std::vector<double>& distances);

		void sortIndicesByDistances(std::vector<long int>& indices, const std::vector<double>& distances);

		void createAngle(Eigen::MatrixX3d& points, Eigen::MatrixX3d& normals, double angle, int nb_points);

		void randomRotation(Eigen::MatrixX3d& pc, Eigen::MatrixX3d& normals);

		void addGaussianNoise(Eigen::MatrixX3d& pc, double sigma);

		void addGaussianNoisePercentage(Eigen::MatrixX3d& pc, int percentage);
};


#endif
