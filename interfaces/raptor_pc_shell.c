/*
 * notes:
 *  - started with
 *  http://www.mcs.anl.gov/petsc/petsc-3.4/src/ksp/ksp/examples/tutorials/ex15.c
 *
 *  - Uses CamelCase to conform to PETSc
 *
 *  - see petsc_raptor_example.c
 *
 *  - TODO: fix docstrings
 *  - TODO: fix 3-space tabs
 *
 */


/*
  Include "petscksp.h" to use KSP solvers.

  automatically includes:
     petscsys.h    - base PETSc routines
     petscvec.h    - vectors
     petscmat.h    - matrices
     petscis.h     - index sets
     petscksp.h    - Krylov subspace methods
     petscviewer.h - viewers
     petscpc.h     - preconditioners
*/
#include <petscksp.h>


#undef __FUNCT__
#define __FUNCT__ "RaptorPCCreate"
/*
   Create a raptor preconditioner context.

   Output Parameter:
    shell - raptor preconditioner context
*/
PetscErrorCode RaptorPCCreate(RaptorPC **shell)
{
  RaptorPC *newctx;
  PetscErrorCode ierr;

  ierr = PetscNew(RaptorPC, &newctx); CHKERRQ(ierr);
  /*
   * TODO initialize newctx here
   */
  *shell       = newctx;
  return 0;
}


#undef __FUNCT__
#define __FUNCT__ "RaptorPCSetUp"
/*
   Set up for  raptor preconditioner context.

   Input Parameters:
   pc    - preconditioner object
   pmat  - preconditioner matrix
   x     - vector  TODO: identify *which* vector

   Output Parameter:
   shell - raptor preconditioner context
*/
PetscErrorCode RaptorPCSetUp(PC pc, Mat pmat, Vec x)
{
  RaptorPC *shell;
  PetscErrorCode ierr;

  ierr = PCShellGetContext(pc, (void**) &shell); CHKERRQ(ierr);
  /*
   * TODO: set things
   */

  return 0;
}


#undef __FUNCT_Raptor
#define __FUNCT__ "RaptorPCApply"
/*
   Run raptor preconditioner

   Input Parameters:
   pc - preconditioner object
   x - input vector

   Output Parameter:
   y - preconditioned vector
*/
PetscErrorCode RaptorPCApply(PC pc, Vec x, Vec y)
{
  RaptorPC *shell;
  PetscErrorCode ierr;

  ierr = PCShellGetContext(pc, (void**) &shell);CHKERRQ(ierr);
  /*
   * TODO: run raptor
   */

  return 0;
}
/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "RaptorPCDestroy"
/*
   Destroy raptor preconditioner context.

   Input Parameter:
   shell - user-defined preconditioner context
*/
PetscErrorCode RaptorPCDestroy(PC pc)
{
  RaptorPC *shell;
  PetscErrorCode ierr;

  ierr = PCShellGetContext(pc, (void**) &shell);CHKERRQ(ierr);
  /*
   * TODO free up the raptor object
   */
  ierr = PetscFree(shell);CHKERRQ(ierr);

  return 0;
}
