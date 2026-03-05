function [dSf_dV1, dSf_dV2, dSt_dV1, dSt_dV2, Sf, St] = dSbr_dV1(Yf, Yt, V, nb, nl, f, t)
%DSBR_DV   Computes partial derivatives of branch power flows w.r.t. voltage.
%   All in polar coordinates edition
%   If = Yf * V;
%   Sf = diag(Vf) * conj(If) = diag(conj(If)) * Vf
%
%   Polar coordinates:
%     Partials of V, Vf & If w.r.t. voltage angles
%       dV/dVa  = j * diag(V)
%       dVf/dVa = sparse(1:nl, f, j * V(f)) = j * sparse(1:nl, f, V(f))
%       dIf/dVa = Yf * dV/dVa = Yf * j * diag(V)
%
%     Partials of V, Vf & If w.r.t. voltage magnitudes
%       dV/dVm  = diag(V./abs(V))
%       dVf/dVm = sparse(1:nl, f, V(f)./abs(V(f))
%       dIf/dVm = Yf * dV/dVm = Yf * diag(V./abs(V))
%
%     Partials of Sf w.r.t. voltage angles
%       dSf/dVa = diag(Vf) * conj(dIf/dVa)
%                       + diag(conj(If)) * dVf/dVa
%               = diag(Vf) * conj(Yf * j * diag(V))
%                       + conj(diag(If)) * j * sparse(1:nl, f, V(f))
%               = -j * diag(Vf) * conj(Yf * diag(V))
%                       + j * conj(diag(If)) * sparse(1:nl, f, V(f))
%               = j * (conj(diag(If)) * sparse(1:nl, f, V(f))
%                       - diag(Vf) * conj(Yf * diag(V)))
%
%     Partials of Sf w.r.t. voltage magnitudes
%       dSf/dVm = diag(Vf) * conj(dIf/dVm)
%                       + diag(conj(If)) * dVf/dVm
%               = diag(Vf) * conj(Yf * diag(V./abs(V)))
%                       + conj(diag(If)) * sparse(1:nl, f, V(f)./abs(V(f)))

%% compute intermediate values
Yfc = conj(Yf);
Ytc = conj(Yt);
Vc = conj(V);
Ifc = Yfc * Vc;     %% conjugate of "from" current
Itc = Ytc * Vc;     %% conjugate of "to" current

diagVf  = sparse(1:nl, 1:nl, V(f), nl, nl);
diagVt  = sparse(1:nl, 1:nl, V(t), nl, nl);
diagIfc = sparse(1:nl, 1:nl, Ifc, nl, nl);
diagItc = sparse(1:nl, 1:nl, Itc, nl, nl);
Vnorm       = V ./ abs(V);
diagVc      = sparse(1:nb, 1:nb, Vc, nb, nb);
diagVnorm   = sparse(1:nb, 1:nb, Vnorm, nb, nb);
CVf  = sparse(1:nl, f, V(f), nl, nb);
CVnf = sparse(1:nl, f, Vnorm(f), nl, nb);
CVt  = sparse(1:nl, t, V(t), nl, nb);
CVnt = sparse(1:nl, t, Vnorm(t), nl, nb);
dSf_dV1 = 1j * (diagIfc * CVf - diagVf * Yfc * diagVc);     %% dSf_dVa
dSf_dV2 = diagVf * conj(Yf * diagVnorm) + diagIfc * CVnf;   %% dSf_dVm
dSt_dV1 = 1j * (diagItc * CVt - diagVt * Ytc * diagVc);     %% dSt_dVa
dSt_dV2 = diagVt * conj(Yt * diagVnorm) + diagItc * CVnt;   %% dSt_dVm
Sf = V(f) .* Ifc;
St = V(t) .* Itc;
end
