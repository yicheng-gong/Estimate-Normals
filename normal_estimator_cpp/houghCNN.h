// Deep Learning for Robust Normal Estimation in Unstructured Point Clouds
// Copyright (c) 2016 Alexande Boulch and Renaud Marlet
//
// This program is free software; you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation; either version 3 of the License, or any later version.
// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
// You should have received a copy of the GNU General Public License along with this
// program; if not, write to the Free Software Foundation, Inc., 51 Franklin Street,
// Fifth Floor, Boston, MA 02110-1301  USA
//
// PLEASE ACKNOWLEDGE THE ORIGINAL AUTHORS AND PUBLICATION:
// "Deep Learning for Robust Normal Estimation in Unstructured Point Clouds "
// by Alexandre Boulch and Renaud Marlet, Symposium of Geometry Processing 2016,
// Computer Graphics Forum

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
		/** \brief Compute Ks
		 *
		 * \input[in] 
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

		/** \brief Compute Ks
		 *
		 * \input[in] 
		 *
		 */
		void 
		getKs(int* array, int m);

		/** \brief MoveIt function for moving arm to pick and  place an object 
		 *
		 * \input[in] desired object  position
		 *
		 * \return true if the action is correct
		 */
		bool 
		getDensitySensitive() {return use_aniso;};

		/** \brief Compute Ks
		 *
		 * \input[in] 
		 *
		 */
		void 
		setDensitySensitive(bool d_s) {use_aniso=d_s;};

		/** \brief Compute the size of Ks 
		 *
		 * \input[in] none
		 *
		 * \return size of Ks
		 */
		int 
		getPCSize() {return _pc.rows();};

		/** \brief Compute the size of Ks 
		 *
		 * \input[in] none
		 *
		 * \return size of Ks
		 */
		int 
		getPCNormalsSize() {return _normals.rows();};

		/** \brief Compute Ks
		 *
		 * \input[in] 
		 *
		 */
		void 
		initialize();

		/** \brief Compute Ks
		 *
		 * \input[in] 
		 *
		 */
		void 
		getBatch(int batch_id, int batch_size, double* array);

		/** \brief Compute Ks
		 *
		 * \input[in] 
		 *
		 */
		void 
		setBatch(int batch_id, int batch_size, double* array);

		/** \brief Compute Ks
		 *
		 * \input[in] 
		 *
		 */
		void 
		setT(int T_) {T=T_;};

		/** \brief Compute Ks
		 *
		 * \input[in] 
		 *
		 */
		void 
		setKaniso(int Kaniso) {K_aniso=Kaniso;};

		/** \brief Compute Ks
		 *
		 * \input[in] 
		 *
		 */
		void 
		setA(int A_){A = A_;}

		/** \brief Compute the size of Ks 
		 *
		 * \input[in] none
		 *
		 * \return size of Ks
		 */
		const int 
		getT()const {return T;}

        /** \brief Compute the size of Ks 
		 *
		 * \input[in] none
		 *
		 * \return size of Ks
		 */
		const int 
		getKaniso() const {return K_aniso;}

		/** \brief Compute the size of Ks 
		 *
		 * \input[in] none
		 *
		 * \return size of Ks
		 */
		const int 
		getA() const{return A;}

		/** \brief Compute Ks
		 *
		 * \input[in] 
		 *
		 */
		void 
		getPoints(double* array, int m, int n);

		/** \brief Compute Ks
		 *
		 * \input[in] 
		 *
		 */
		void 
		getNormals(double* array, int m, int n);

		/** \brief Compute Ks
		 *
		 * \input[in] 
		 *
		 */
		void 
		setPoints(double* array, int m, int n);

		/** \brief Compute Ks
		 *
		 * \input[in] 
		 *
		 */
		void 
		setNormals(double* array, int m, int n);

		/** \brief Compute the size of Ks 
		 *
		 * \input[in] none
		 *
		 * \return size of Ks
		 */
		const MatrixX3& 
		pc() const{return _pc;}

        /** \brief Compute the size of Ks 
		 *
		 * \input[in] none
		 *
		 * \return size of Ks
		 */
		MatrixX3& 
		normals(){return _normals;}

		/** \brief Compute the size of Ks 
		 *
		 * \input[in] none
		 *
		 * \return size of Ks
		 */
		int 
		generateTrainAccRandomCorner(int noise_val, int n_points, double* array, double* array_gt);
		
		/** \brief Compute Ks
		 *
		 * \input[in] 
		 *
		 */
		void 
		loadXYZ(const std::string& filename);

		
		/** \brief Compute Ks
		 *
		 * \input[in] 
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

		VectorX gaussianBlur(VectorX& accum, int kernelSize, float sigma, int A = 33);
};




#endif
