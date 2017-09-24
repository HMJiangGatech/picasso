#ifndef PICASSO_ACTNEWTON_H
#define PICASSO_ACTNEWTON_H

#include <cmath>
#include <string>

#include <picasso/solver_params.h>

namespace picasso {
namespace solver {
class ActNewtonSolver {
private:
  const PicassoSolverParams m_param;
  const ObjFunction *m_obj;

  std::vector<int> itercnt_path;
  std::vector<ModelParam> solution_path;

public:
  ActNewtonSolver(ObjFunction *obj, PicassoSolverParams param)
      : m_param(param), m_obj(obj);

  void solve(ObjFunction *obj);
};

} // namespace solver
} // namespace picasso

#endif // PICASSO_ACTNEWTON_H