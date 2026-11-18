"""Generate architecture_figure.tex from one set of spacing constants.

Scheme: every labelled arrow is G long, every unlabelled arrow g long, rows and
stacked items are V apart; cuboid geometry follows \\mdfmap (depth 0.30h, 0.22h).
"""
import sys

G = 1.80    # labelled arrow
g = 0.50    # unlabelled arrow
V = 0.40    # vertical clearance between rows / items
P = 0.30    # padding inside frames
hL, hH = 1.45, 2.35
CH = 0.30   # height of a channel label above a cuboid
SHEET = 0.13  # upward offset of the stacked sheets behind the per-band frame
              # (straight up, so the arrows in and out cross one edge only)
RES_OFF = 0.11  # offset of the stacked residual boxes
PILL_IN_W, PILL_OUT_W, PILL_H = 2.70, 2.40, 1.00
RELU_W = 1.50
LRS = 1.10  # input thumbnail side; output is twice this
STACK = 0.21  # \mdimstack back planes

def W(t, h):            # total width of a cuboid
    return t + 0.30 * h
def TOP(h):             # top of a cuboid above its centre line
    return 0.5 * h + 0.22 * h

out = []
o = out.append
f = lambda v: f"{v:.3f}"

def fmap(name, x, y, t, h, col, slabs=None, ch=None):
    o(f"\\mdfmap{{{name}}}{{{f(x)}}}{{{f(y)}}}{{{t}}}{{{h}}}{{{col}}}")
    if slabs:
        o(f"\\mdslabs{{{f(x)}}}{{{f(y)}}}{{{t}}}{{{h}}}{{{slabs}}}{{{col}}}")
    if ch is not None:
        o(f"\\node[mdch,above=1pt of {name}-n] {{{ch}}};")
    return x + W(t, h)

def arrow(a, b, label=None, where="above"):
    lab = f" node[mdop,midway,{where}=1pt] {{{label}}}" if label else ""
    o(f"\\draw[mdflow] {a} -- {b}{lab};")

# ------------------------------------------------------------------ (a) trunk
o("% ---- shared trunk (y = 0); input column on the left")
x = 0.0
xin = PILL_IN_W / 2
o(f"\\node[mdfun,minimum width={PILL_IN_W}cm,minimum height={PILL_H}cm] (asinh) at ({f(xin)},0)"
  " {\\mdgasinh\\ \\mdtwo{$\\mathrm{arcsinh}(x/k_j)$}{8 knees}};")
xS = PILL_IN_W + g
eS = fmap("S", xS, 0, 0.72, hL, "mdLR", slabs=8, ch="32")
arrow("(asinh.east)", "(S-w)")
xF = eS + G
eF = fmap("F", xF, 0, 0.72, hL, "mdLR", ch="32")
arrow("(S-e)", "(F-w)", "$3\\!\\times\\!3$ conv")
res_w = 1.90
res_x0 = eF + g
res_x1 = res_x0 + res_w
o("\\foreach \\k in {2,1}{%")
o(f"  \\draw[rounded corners=2pt,line width=0.6pt,draw=mdRes!75!black,fill=mdRes!22]"
  f" ($({f(res_x0)},{f(-hL/2)})+({RES_OFF}*\\k,{RES_OFF}*\\k)$) rectangle ++({res_w},{hL});}}")
o(f"\\node[mdbox,draw=mdRes!75!black,fill=mdRes!38,minimum width={res_w}cm,minimum height={hL}cm]"
  f" (res) at ({f((res_x0+res_x1)/2)},0) {{16 residual\\\\ blocks}};")
arrow("(F-e)", "(res.west)")
res_e = res_x1 + 2 * RES_OFF
xT = res_e + G
eT = fmap("T", xT, 0, 0.59, hL, "mdLR", ch="16")
arrow(f"({f(res_e)},0)", "(T-w)", "$3\\!\\times\\!3$ conv")
xM = eT + G
eM = fmap("M", xM, 0, 0.33, hH, "mdHR", ch="4")
arrow("(T-e)", "(M-w)", "upsample")
xs = eM + g + 0.21
o(f"\\node[mdsum] (sum) at ({f(xs)},0) {{$+$}};")
arrow("(M-e)", "(sum)")

# input image under the input function box
in_top = -PILL_H / 2 - g                    # top of the image stack
yi = in_top - STACK - LRS / 2
o(f"\\mdimstack{{in}}{{{f(xin - LRS/2 - 0.105)}}}{{{f(yi)}}}{{{LRS}}}{{lr_rgb}}")
o(f"\\draw[mdflow] ({f(xin)},{f(in_top)}) -- (asinh.south);")
o(f"\\node[mdop,anchor=north] (inlab) at ({f(xin)},{f(yi - LRS/2 - 0.06)})"
  " {Euclid image $x$\\\\[-1pt] 4 bands, $0.1''$ grid};")

# output column under the sum
pill_out_top = -TOP(hH) - V
yk = pill_out_top - PILL_H / 2
o(f"\\node[mdfun,minimum width={PILL_OUT_W}cm,minimum height={PILL_H}cm] (knee) at ({f(xs)},{f(yk)})"
  " {\\mdgsinh\\ \\mdtwo{$k_b\\sinh y_b$}{learned $k_b$}};")
o("\\draw[mdflow] (sum) -- (knee.north) node[mdop,midway,right=1pt] {$y_b$};")
out_top = yk - PILL_H / 2 - g
yo = out_top - STACK - LRS
o(f"\\mdimstack{{out}}{{{f(xs - LRS - 0.105)}}}{{{f(yo)}}}{{{2*LRS}}}{{sr_rgb}}")
o(f"\\draw[mdflow] (knee.south) -- ({f(xs)},{f(out_top)});")
o(f"\\node[mdop,anchor=north] at ({f(xs)},{f(yo - LRS - 0.06)})"
  " {reconstruction\\\\[-1pt] 4 bands, $0.05''$ grid};")

# ------------------------------------------------------------------ (a) per-band row
ys = TOP(hH) + CH + V + hH / 2                  # its tall cuboids clear M's label
o("% ---- per-band path, drawn as four stacked sheets")
xKA = xM
frame_e = xKA - G                              # frame's right edge
eKS = frame_e - P
xKS = eKS - W(0.07, hH)
xKC = xKS - G - W(0.33, hL)
xK = xKC - G - W(0.46, hL)
frame_w = xK - P
frame_b = ys - hH / 2 - P
frame_t = ys + TOP(hH) + CH + P
for k in (3, 2, 1):
    o(f"\\draw[rounded corners=3pt,line width=0.5pt,draw=black!30,fill=white]"
      f" ({f(frame_w)},{f(frame_b + SHEET*k)}) rectangle ({f(frame_e)},{f(frame_t + SHEET*k)});")
o(f"\\draw[rounded corners=3pt,line width=0.6pt,draw=black!55,dashed,fill=white]"
  f" ({f(frame_w)},{f(frame_b)}) rectangle ({f(frame_e)},{f(frame_t)});")
o(f"\\node[mdnote,anchor=north west] at ({f(frame_w + 0.12)},{f(frame_t - 0.10)}) {{each band separately}};")
fmap("K", xK, ys, 0.46, hL, "mdLR", slabs=8, ch="8")
fmap("KC", xKC, ys, 0.33, hL, "mdLR", ch="4")
fmap("KS", xKS, ys, 0.07, hH, "mdHR", ch="1")
fmap("KA", xKA, ys, 0.33, hH, "mdHR", ch="4")
o(f"\\draw[mdflow,rounded corners=5pt] (asinh.north) |- (K-w);")
arrow("(K-e)", "(KC-w)", "$5\\!\\times\\!5$ conv")
arrow("(KC-e)", "(KS-w)", "upsample")
arrow("(KS-e)", "(KA-w)")
o(f"\\node[mdop,above=1pt] at ({f((frame_e + xKA)/2)},{f(ys)}) {{stack}};")
o("\\draw[mdflow,rounded corners=5pt] (KA-e) -| (sum.north);")

# legend: in the corner between the per-band branch and the trunk
o("% ---- legend")
xl = xin + 0.45
for i, (kind, text) in enumerate([("LR", "$0.1''$ grid"), ("HR", "$0.05''$ grid"),
                                  ("fun", "non-linear function")]):
    yl = ys - 0.55 - 0.60 * i
    if kind == "fun":
        o(f"\\node[mdfun,inner sep=1.5pt] at ({f(xl + 0.18)},{f(yl)}) {{\\mdgrelu}};")
    else:
        o(f"\\mdfmap{{lg{i}}}{{{f(xl + 0.05)}}}{{{f(yl)}}}{{0.22}}{{0.42}}{{md{kind}}}")
    o(f"\\node[mdleg,anchor=west] at ({f(xl + 0.50)},{f(yl)}) {{{text}}};")
o(f"\\node[mdpanel,anchor=north west] at (0,{f(frame_t + 3 * SHEET)}) {{(a)}};")

# ------------------------------------------------------------------ (b)
o("% ===================================================================== (b)")
bT = yi - LRS / 2 - 0.06 - 0.80 - V          # below the input label
bL = 0.50
title_h = 0.40
yb = bT - P - title_h - CH - TOP(hL)
x = bL + P
ebx = fmap("bx", x, yb, 0.72, hL, "mdLR", ch="32")
eb1 = fmap("b1", ebx + G, yb, 1.06, hL, "mdLR", ch="192")
relu_x = eb1 + g + RELU_W / 2
o(f"\\node[mdfun,minimum width={RELU_W}cm,minimum height={PILL_H*0.8}cm] (relu) at ({f(relu_x)},{f(yb)}) {{\\mdgrelu\\ ReLU}};")
eb2 = fmap("b2", relu_x + RELU_W / 2 + G, yb, 0.67, hL, "mdLR", ch="25")
eb3 = fmap("b3", eb2 + G, yb, 0.72, hL, "mdLR", ch="32")
bsum = eb3 + g + 0.21
o(f"\\node[mdsum] (bsum) at ({f(bsum)},{f(yb)}) {{$+$}};")
bR = bsum + 0.21 + g
arrow("(bx-e)", "(b1-w)", "$1\\!\\times\\!1$ conv")
arrow("(b1-e)", "(relu.west)")
arrow("(relu.east)", "(b2-w)", "$1\\!\\times\\!1$ conv")
arrow("(b2-e)", "(b3-w)", "$3\\!\\times\\!3$ conv")
arrow("(b3-e)", "(bsum)")
arrow("(bsum)", f"({f(bR)},{f(yb)})")
yid = yb - hL / 2 - 0.55
o(f"\\draw[mdflow,rounded corners=5pt] ($(bx-s)+(0.06,0)$) |- ({f(bsum)},{f(yid)}) -- (bsum.south);")
o(f"\\node[mdop,above=1pt] at ({f(relu_x + RELU_W/2 + G/2)},{f(yid)}) {{identity}};")
bB = yid - P
o(f"\\begin{{scope}}[on background layer]")
o(f"  \\fill[mdRes!12] (res.south west) -- (res.south east) -- ({f(bR - 2.5)},{f(bT)}) -- ({f(bL + 3.0)},{f(bT)}) -- cycle;")
o("\\end{scope}")
o(f"\\draw[mdRes!70!black,line width=0.5pt] (res.south west) -- ({f(bL + 3.0)},{f(bT)}) (res.south east) -- ({f(bR - 2.5)},{f(bT)});")
o(f"\\begin{{scope}}[on background layer]\\draw[rounded corners=4pt,draw=mdRes!85!black,line width=0.9pt,fill=white] ({f(bL)},{f(bB)}) rectangle ({f(bR)},{f(bT)});\\end{{scope}}")
o(f"\\node[font=\\fontsize{{9}}{{10.5}}\\selectfont\\bfseries,text=mdRes!65!black,anchor=north west,inner sep=0pt]"
  f" at ({f(bL + P)},{f(bT - 0.15)}) {{one residual block}};")
o(f"\\node[mdpanel,anchor=north east] at ({f(bL - 0.08)},{f(bT)}) {{(b)}};")

sys.stderr.write(f"sum x={xs:.2f}  out pill left={xs-PILL_OUT_W/2:.2f}  (b) right={bR:.2f}  "
                 f"frame_w={frame_w:.2f}  K={xK:.2f}  top={frame_t+3*SHEET:.2f}  bottom={bB:.2f}\n")

body = "\n".join(out)
head = r"""% Architecture figure: the whole figure* block (tikzpicture + caption).
% GENERATED by paper_figures/overleaf_model_stars/architecture_gen.py from one
% set of spacing constants (labelled arrows G, unlabelled g, clearance V);
% edit the script, not the coordinates.
% Needs architecture_preamble.tex and images/thumbs/{lr,sr}_rgb.png.
% No blank lines inside the tikzpicture: with the linenumbers class option each
% one would end the build with "Improper \prevdepth".
\begin{figure*}
\centering
\begin{tikzpicture}
"""
tail = r"""
\end{tikzpicture}
\caption{The reconstruction network.
(a) The four-band Euclid image $x$ (electrons, $0.1''$ pixels) is stretched with $\mathrm{arcsinh}(x/k_j)$ at eight knees $k_j = 10^{-3}, 10^{-2}, \ldots, 10^{4}$ electrons, giving $4\times8=32$ channels.
Two paths start from this stack.
The shared trunk (middle row) mixes all bands: a $3\times3$ convolution, $16$ residual blocks, a second $3\times3$ convolution and a $2\times$ upsampling, which rearranges each group of four channels into a $2\times2$ patch of pixels on the $0.05''$ grid.
The per-band path (top row) upsamples each band from its own eight channels with a $5\times5$ convolution, so it never mixes bands.
The two are added, giving the stretched image $y_b$ for each band $b$, and $k_b \sinh y_b$ with a learned knee $k_b$ turns it back into electrons.
Gray cuboids are feature maps on the $0.1''$ grid and green ones on the $0.05''$ grid; numbers above them are channel counts.
Purple boxes are the only non-linear functions; every other step is linear.
(b) One of the $16$ residual blocks: a $1\times1$ convolution widens $32$ channels to $192$, a ReLU, a $1\times1$ convolution narrows them to $25$, and a $3\times3$ convolution returns $32$ channels, which are added to the block input.
The thumbnails show a Q1 lens candidate and its reconstruction (R\,=\,$H_{\rm E}$, G\,=\,mean of $Y_{\rm E}$ and $J_{\rm E}$, B\,=\,VIS).}
\label{fig:architecture}
\end{figure*}
"""
open(sys.argv[1], "w").write(head + body + tail)
