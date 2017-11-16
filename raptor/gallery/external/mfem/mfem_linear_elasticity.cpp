#include "gallery/external/mfem_wrapper.hpp"

using namespace mfem;

// Create an MFEM Linear Elasticity Matrix and convert to Raptor format
raptor::ParCSRMatrix* mfem_linear_elasticity(raptor::ParVector& x_raptor, 
        raptor::ParVector& b_raptor,
        const char* mesh_file, int order, int seq_n_refines, 
        int par_n_refines, MPI_Comm comm)
{
    int rank, num_procs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_procs);

    int mesh_dim;
    int par_mesh_n;
    int boundary_n;

    Mesh* mesh;
    ParMesh* par_mesh;
    FiniteElementCollection* collection;
    ParFiniteElementSpace* space;

    mesh = new Mesh(mesh_file, 1, 1);
    mesh_dim = mesh->Dimension();

    // Uniform refinement on serial mesh
    for (int i = 0; i < seq_n_refines; i++)
    {
        mesh->UniformRefinement();
    }

    // Uniform refinement on parallel mesh
    par_mesh = new ParMesh(comm, *mesh);
    delete mesh;
    for (int i = 0; i < par_n_refines; i++)
    {
        par_mesh->UniformRefinement();
    }

    // Get dims
    par_mesh_n = par_mesh->attributes.Max();
    boundary_n = par_mesh->bdr_attributes.Max();

    // Form finite element collection / space
    collection = new H1_FECollection(order, mesh_dim);
    space = new ParFiniteElementSpace(par_mesh, collection, mesh_dim, Ordering::byVDIM);

    Array<int> dofs;
    Array<int> bdry(boundary_n);
    bdry = 0;
    bdry[0] = 1;
    space->GetEssentialTrueDofs(bdry, dofs);

    VectorArrayCoefficient force(mesh_dim);
    for (int i = 0; i < mesh_dim-1; i++)
    {
        force.Set(i, new ConstantCoefficient(0.0));
    }
    mfem::Vector pull_force(boundary_n);
    pull_force = 0.0;
    pull_force(1) = -1.0e-2;
    force.Set(mesh_dim-1, new PWConstCoefficient(pull_force));

    ParLinearForm* b = new ParLinearForm(space);
    b->AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(force));
    b->Assemble();

    ParGridFunction x(space);
    x = 0.0;

    mfem::Vector lambda(par_mesh_n);
    lambda = 1.0;
    lambda(0) = lambda(1) * 50;
    PWConstCoefficient lambda_func(lambda);
    mfem::Vector mu(par_mesh_n);
    mu = 1.0;
    mu(0) = mu(1) * 50;
    PWConstCoefficient mu_func(mu);

    ParBilinearForm *a = new ParBilinearForm(space);
    a->AddDomainIntegrator(new ElasticityIntegrator(lambda_func, mu_func));
    a->EnableStaticCondensation();
    a->Assemble();

    HypreParMatrix A;
    mfem::Vector B, X;
    a->FormLinearSystem(dofs, x, *b, A, X, B);

    A.SetOwnerFlags(-1, -1, -1);
    hypre_ParCSRMatrix* A_hypre = A.StealData();

    raptor::ParCSRMatrix *A_raptor = convert(A_hypre, comm);
    x_raptor.resize(A_raptor->global_num_rows, A_raptor->local_num_rows, 
            A_raptor->partition->first_local_row);
    b_raptor.resize(A_raptor->global_num_rows, A_raptor->local_num_rows, 
            A_raptor->partition->first_local_row);
    double* x_data = X.GetData();
    double* b_data = B.GetData();
    for (int i = 0; i < A_raptor->local_num_rows; i++)
    {
        x_raptor[i] = x_data[i];
        b_raptor[i] = b_data[i];
    }

    delete a;
    delete b;
   
    delete space;
    delete collection;
    delete par_mesh;

    return A_raptor;
}
