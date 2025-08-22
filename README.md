# Local Minimax Note

Just some ideas I've been messing around with

This repo is just a rough write-up (in LaTeX) of an idea for a first-order method  
that actually converges to **local minimax optima** in nonconvex–nonconcave games.  

Why? Because training GANs is still painful, and most gradient methods either  
cycle, collapse, or land on the wrong kind of point. The algorithm here  
("VR-TTEG" — Vanishing Regularization Two-Timescale Extragradient) is a simple tweak  
that uses timescale separation, extragradient steps, and a vanishing regularizer.  

The `main.tex` file compiles into a 6–8 page note with proofs, constants,  
and toy counterexamples showing why each ingredient matters.  


This Python script is a toy experimental framework for testing whether the proposed algorithm (VR-TTEG: Vanishing Regularization Two-Timescale Extragradient) actually converges to local minimax optima in simple adversarial games. It implements the algorithm with tunable options (extragradient on/off, timescale separation on/off, regularization on/off), runs it on three toy payoff functions (including a non-strict local minimax case), and numerically checks whether the final point satisfies the local minimax conditions. By comparing the full algorithm against ablations, you can see why each ingredient is necessary: without timescale separation the dynamics cycle, without extragradient they drift to saddles, and without vanishing regularization the inner maximization is ill-posed. Plots of the trajectories make these behaviors visually clear.
Compile with:
```bash
pdflatex main.tex
