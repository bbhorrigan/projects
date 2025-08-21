# Local Minimax Note

This repo is just a rough write-up (in LaTeX) of an idea for a first-order method  
that actually converges to **local minimax optima** in nonconvex–nonconcave games.  

Why? Because training GANs is still painful, and most gradient methods either  
cycle, collapse, or land on the wrong kind of point. The algorithm here  
("VR-TTEG" — Vanishing Regularization Two-Timescale Extragradient) is a simple tweak  
that uses timescale separation, extragradient steps, and a vanishing regularizer.  

The `main.tex` file compiles into a 6–8 page note with proofs, constants,  
and toy counterexamples showing why each ingredient matters.  

Compile with:
```bash
pdflatex main.tex
