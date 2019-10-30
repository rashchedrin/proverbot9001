(* This program is free software; you can redistribute it and/or      *)
(* modify it under the terms of the GNU Lesser General Public License *)
(* as published by the Free Software Foundation; either version 2.1   *)
(* of the License, or (at your option) any later version.             *)
(*                                                                    *)
(* This program is distributed in the hope that it will be useful,    *)
(* but WITHOUT ANY WARRANTY; without even the implied warranty of     *)
(* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the      *)
(* GNU General Public License for more details.                       *)
(*                                                                    *)
(* You should have received a copy of the GNU Lesser General Public   *)
(* License along with this program; if not, write to the Free         *)
(* Software Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA *)
(* 02110-1301 USA                                                     *)


Require Import Reals.

(* Large approximation of PI *)
Definition PI_lb : R := 3%R.
Definition PI_ub : R := 4%R.

(* 3 < PI < 4 *)
Lemma PI_approx : (PI_lb < PI < PI_ub)%R.
Proof with trivial.
split...
elim (PI_ineq 3); intros H0 _;
 cut (sum_f_R0 (tg_alt PI_tg) (S (2 * 3)) = (33976 / 45045)%R)...
intro; rewrite H in H0; apply Rmult_lt_reg_l with (/ 4)%R...
apply Rinv_0_lt_compat; prove_sup...
unfold PI_lb in |- *; simpl in |- *; rewrite <- (Rmult_comm PI);
 apply Rlt_le_trans with (33976 / 45045)%R...
apply Rmult_lt_reg_l with 4%R...
prove_sup...
rewrite <- Rmult_assoc; rewrite <- Rinv_r_sym...
rewrite Rmult_1_l; apply Rmult_lt_reg_l with 45045%R...
prove_sup...
pattern 45045%R at 1 in |- *; rewrite <- Rmult_comm; unfold Rdiv in |- *;
 repeat rewrite Rmult_assoc; rewrite <- Rinv_l_sym...
prove_sup...
discrR...
discrR...
unfold tg_alt, PI_tg in |- *;
 replace
  (sum_f_R0 (fun i : nat => ((-1) ^ i * / INR (2 * i + 1))%R) (S (2 * 3)))
  with (1 - / 3 + / 5 - / 7 + / 9 - / 11 + / 13 - / 15)%R...
assert (H : 45045%R <> 0%R)...
discrR...
apply Rmult_eq_reg_l with 45045%R...
unfold Rdiv in |- *; rewrite <- (Rmult_comm (/ 45045));
 rewrite <- (Rmult_assoc 45045 (/ 45045) 33976); rewrite <- Rinv_r_sym...
rewrite Rmult_1_l; replace 45045%R with (3 * 3 * 5 * 7 * 11 * 13)%R;
 [ idtac | Rcompute ]...
replace 15%R with (3 * 5)%R; [ idtac | Rcompute ]...
replace 9%R with (3 * 3)%R; [ idtac | Rcompute ]...
repeat rewrite Rinv_mult_distr; try discrR...
set (x := 13%R); set (y := 11%R); set (z := 7%R); set (t := 5%R);
 set (u := 3%R);
 replace
  (u * u * t * z * y * x *
   (1 - / u + / t - / z + / u * / u - / y + / x - / u * / t))%R with
  (u * u * t * z * y * x * 1 - u * t * z * y * x * (u * / u) +
   u * u * z * y * x * (t * / t) - u * u * t * y * x * (z * / z) +
   t * z * y * x * (u * / u) * (u * / u) - u * u * t * z * x * (y * / y) +
   u * u * t * z * y * (x * / x) - u * z * y * x * (u * / u) * (t * / t))%R;
 [ idtac | ring ]...
repeat rewrite <- Rinv_r_sym; try unfold x, y, z, t, u in |- *; discrR...
repeat rewrite Rmult_1_r; Rcompute...
simpl in |- *; rewrite Rinv_1; repeat rewrite Rmult_1_r;
 unfold Rminus in |- *...
replace (1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1)%R with 15%R;
 [ idtac | ring ]...
replace (1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1)%R with 13%R;
 [ idtac | ring ]...
replace (1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1)%R with 11%R; [ idtac | ring ]...
replace (1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1)%R with 9%R; [ idtac | ring ]...
replace (1 + 1 + 1 + 1 + 1 + 1 + 1)%R with 7%R; [ idtac | ring ]...
replace (1 + 1 + 1 + 1 + 1)%R with 5%R; [ idtac | ring ]...
replace (1 + 1 + 1)%R with 3%R; [ idtac | rewrite Rplus_comm ]...
change (-1)%R with (Ropp R1).
repeat rewrite Rplus_assoc; apply Rplus_eq_compat_l...
repeat (rewrite Rmult_opp_opp; rewrite Rmult_1_r); repeat rewrite Rmult_1_r;
 repeat (rewrite Rmult_opp_opp; rewrite Rmult_1_r); 
 repeat rewrite Rmult_1_r; repeat (rewrite Rmult_opp_opp; rewrite Rmult_1_r);
 repeat rewrite Rmult_1_r; repeat rewrite Rmult_1_l;
 repeat rewrite Ropp_mult_distr_l_reverse; repeat rewrite Rmult_1_l...
elim (PI_ineq 1); intros _ H0; unfold tg_alt, PI_tg in H0; simpl in H0;
 rewrite Rinv_1 in H0; repeat rewrite Rmult_1_r in H0...
cut ((1 + 1 + 1 + 1 + 1)%R = 5%R); [ intro; rewrite H in H0; clear H | ring ]...
cut ((1 + 1 + 1)%R = 3%R); [ intro; rewrite H in H0; clear H | ring ]...
change (-1)%R with (Ropp R1) in H0.
rewrite Rmult_opp_opp in H0; repeat rewrite Rmult_1_l in H0;
 apply Rmult_lt_reg_l with (/ 4)%R...
apply Rinv_0_lt_compat; prove_sup...
rewrite <- (Rmult_comm PI); apply Rle_lt_trans with (1 + -1 * / 3 + / 5)%R...
unfold PI_ub in |- *; simpl in |- *; rewrite <- Rinv_l_sym;
 [ idtac | discrR ]...
change (-1)%R with (Ropp R1) in *.
pattern 1%R at 2; rewrite <- Rplus_0_r; repeat rewrite Rplus_assoc;
 apply Rplus_lt_compat_l; rewrite Ropp_mult_distr_l_reverse;
 rewrite Rmult_1_l; apply Rplus_lt_reg_l with (/ 3)%R; 
 rewrite Rplus_0_r; rewrite <- Rplus_assoc; rewrite Rplus_opp_r;
 rewrite Rplus_0_l; apply Rinv_lt_contravar; prove_sup...
Qed.