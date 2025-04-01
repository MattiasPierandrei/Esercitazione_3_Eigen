//	PIERANDREI MATTIAS

#include <iostream>
#include <iomanip>
#include "Eigen/Eigen"

using namespace std;
using namespace Eigen;

VectorXd s = VectorXd::Constant(2, -1); // Exact solution [-1, -1]

// Function to compute the relative error
void ERROR(const VectorXd& x)
{
    cout << "\nRelative Error = " << ((s - x).norm() / s.norm()) << endl;
}

// PALU Decomposition
void PALU(const MatrixXd& A , const VectorXd& b)
{
    PartialPivLU<MatrixXd> lu(A);
    VectorXd x = lu.solve(b); 
    cout << "\nAx = b with PALU:\n" << x << endl;
    ERROR(x);
}

// QR Decomposition
void QR(const MatrixXd& A , const VectorXd& b)
{
    HouseholderQR<MatrixXd> qr(A);
    VectorXd x = qr.solve(b); 
    cout << "\nAx = b with QR:\n" << x << endl;
    ERROR(x);
}

int main()
{
    cout << scientific << setprecision(16) << endl;
	
	/////////////////////////////////////////////////////////////////////////////////////////
    
    cout << "\n[1]----------------------------------" << endl;
    
    MatrixXd A(2,2);
    A << 5.547001962252291e-01, -3.770900990025203e-02,
         8.320502943378437e-01, -9.992887623566787e-01;
    cout << "Matrix A:\n" << A << endl;

    VectorXd b(2);  
    b << -5.169911863249772e-01, 1.672384680188350e-01;
    cout << "Vector b:\n" << b << endl;

    PALU(A, b);
    QR(A, b);
    
    cout << "---------------------------------------" << endl;
	
	/////////////////////////////////////////////////////////////////////////////////////////
	
	cout << "\n[2]----------------------------------" << endl;
    
    A << 5.547001962252291e-01, -5.540607316466765e-01,
		 8.320502943378437e-01, -8.324762492991313e-01;
    cout << "Matrix A:\n" << A << endl;
  
    b << -6.394645785530173e-04 , 4.259549612877223e-04;
    cout << "Vector b:\n" << b << endl;

    PALU(A, b);
    QR(A, b);
    
    cout << "---------------------------------------" << endl;
	
	/////////////////////////////////////////////////////////////////////////////////////////
	
	cout << "\n[3]----------------------------------" << endl;
    
    A << 5.547001962252291e-01, -5.547001955851905e-01 ,
		 8.320502943378437e-01, -8.320502947645361e-01;
    cout << "Matrix A:\n" << A << endl;
 
    b << -6.400391328043042e-10 , 4.266924591433963e-10;
    cout << "Vector b:\n" << b << endl;

    PALU(A, b);
    QR(A, b);
    
    cout << "---------------------------------------" << endl;

    return 0;
}
