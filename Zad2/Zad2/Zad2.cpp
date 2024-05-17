#include <mpi.h>
#include <iostream>
#include <vector>

// Функция для ввода матрицы с консоли
void readMatrixFromConsole(std::vector<std::vector<double>>& matrix, int& n) {
    std::cout << "Enter the size of the matrix (n): ";
    std::cin >> n;
    matrix.resize(n, std::vector<double>(n));
    std::cout << "Enter the elements of the matrix (row-wise):\n";
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cin >> matrix[i][j];
        }
    }
}

// Функция для вывода матрицы
void printMatrix(const std::vector<std::vector<double>>& matrix) {
    for (const auto& row : matrix) {
        for (const auto& elem : row) {
            std::cout << elem << " ";
        }
        std::cout << std::endl;
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n;
    std::vector<std::vector<double>> a;

    if (rank == 0) {
        // Считываем матрицу с консоли
        readMatrixFromConsole(a, n);
    }

    // Передаем размер матрицы всем процессам
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Распределение матрицы между процессами
    int rows_per_process = n / size;
    int remaining_rows = n % size;

    std::vector<std::vector<double>> local_matrix(rows_per_process + (rank < remaining_rows ? 1 : 0), std::vector<double>(n));

    // Подготовка для отправки/приема данных
    std::vector<int> sendcounts(size);
    std::vector<int> displs(size);

    int offset = 0;
    for (int i = 0; i < size; ++i) {
        sendcounts[i] = (rows_per_process + (i < remaining_rows ? 1 : 0)) * n;
        displs[i] = offset;
        offset += sendcounts[i];
    }

    MPI_Scatterv(rank == 0 ? &a[0][0] : nullptr, sendcounts.data(), displs.data(), MPI_DOUBLE, &local_matrix[0][0], sendcounts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Вычисление средних арифметических соседей для локальной матрицы
    std::vector<std::vector<double>> local_result = local_matrix;

    for (int i = 1; i < local_matrix.size() - 1; ++i) {
        for (int j = 1; j < n - 1; ++j) {
            local_result[i][j] = (local_matrix[i - 1][j] + local_matrix[i + 1][j] + local_matrix[i][j - 1] + local_matrix[i][j + 1]) / 4.0;
        }
    }

    // Сбор данных
    MPI_Gatherv(&local_result[0][0], sendcounts[rank], MPI_DOUBLE, rank == 0 ? &a[0][0] : nullptr, sendcounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "The resulting matrix is:\n";
        printMatrix(a);
    }

    MPI_Finalize();
    return 0;
}