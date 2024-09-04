import multiple_linear_regression
import polynomial_regression
import support_vector_regression
import decision_tree_regression
import random_forest_regression

score_multi = multiple_linear_regression.m_score.round(5) # type: ignore
score_multi2 = multiple_linear_regression.score2.round(5) # type: ignore
score_multi3 = multiple_linear_regression.score3.round(5) # type: ignore

score_poly = polynomial_regression.score.round(5) # type: ignore
score_poly2 = polynomial_regression.score2.round(5) # type: ignore
score_poly3 = polynomial_regression.score3.round(5) #type: ignore


score_svr = support_vector_regression.sv_score.round(5) # type: ignore
score_svr2 = support_vector_regression.score2.round(5) # type: ignore
score_svr3 = support_vector_regression.score3.round(5) # type: ignore

score_tree = decision_tree_regression.score.round(5) # type: ignore
score_tree2 = decision_tree_regression.score2.round(5) # type: ignore
score_tree3 = decision_tree_regression.score3.round(5) # type: ignore

score_forest = random_forest_regression.score.round(5) # type: ignore
score_forest2 = random_forest_regression.score2.round(5) # type: ignore
score_forest3 = random_forest_regression.score3.round(5) # type: ignore