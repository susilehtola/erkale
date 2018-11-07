r_H=load("r_H.dat");
v_H=gszpot(1,r_H);
out=fopen("v_H.dat","w");
for i=1:length(r_H)
  fprintf(out,"%.15e %.15e\n",r_H(i),v_H(i));
end
fclose(out);
r_He=load("r_He.dat");
v_He=gszpot(2,r_He);
out=fopen("v_He.dat","w");
for i=1:length(r_He)
  fprintf(out,"%.15e %.15e\n",r_He(i),v_He(i));
end
fclose(out);
r_Li=load("r_Li.dat");
v_Li=gszpot(3,r_Li);
out=fopen("v_Li.dat","w");
for i=1:length(r_Li)
  fprintf(out,"%.15e %.15e\n",r_Li(i),v_Li(i));
end
fclose(out);
r_Be=load("r_Be.dat");
v_Be=gszpot(4,r_Be);
out=fopen("v_Be.dat","w");
for i=1:length(r_Be)
  fprintf(out,"%.15e %.15e\n",r_Be(i),v_Be(i));
end
fclose(out);
r_B=load("r_B.dat");
v_B=gszpot(5,r_B);
out=fopen("v_B.dat","w");
for i=1:length(r_B)
  fprintf(out,"%.15e %.15e\n",r_B(i),v_B(i));
end
fclose(out);
r_C=load("r_C.dat");
v_C=gszpot(6,r_C);
out=fopen("v_C.dat","w");
for i=1:length(r_C)
  fprintf(out,"%.15e %.15e\n",r_C(i),v_C(i));
end
fclose(out);
r_N=load("r_N.dat");
v_N=gszpot(7,r_N);
out=fopen("v_N.dat","w");
for i=1:length(r_N)
  fprintf(out,"%.15e %.15e\n",r_N(i),v_N(i));
end
fclose(out);
r_O=load("r_O.dat");
v_O=gszpot(8,r_O);
out=fopen("v_O.dat","w");
for i=1:length(r_O)
  fprintf(out,"%.15e %.15e\n",r_O(i),v_O(i));
end
fclose(out);
r_F=load("r_F.dat");
v_F=gszpot(9,r_F);
out=fopen("v_F.dat","w");
for i=1:length(r_F)
  fprintf(out,"%.15e %.15e\n",r_F(i),v_F(i));
end
fclose(out);
r_Ne=load("r_Ne.dat");
v_Ne=gszpot(10,r_Ne);
out=fopen("v_Ne.dat","w");
for i=1:length(r_Ne)
  fprintf(out,"%.15e %.15e\n",r_Ne(i),v_Ne(i));
end
fclose(out);
r_Na=load("r_Na.dat");
v_Na=gszpot(11,r_Na);
out=fopen("v_Na.dat","w");
for i=1:length(r_Na)
  fprintf(out,"%.15e %.15e\n",r_Na(i),v_Na(i));
end
fclose(out);
r_Mg=load("r_Mg.dat");
v_Mg=gszpot(12,r_Mg);
out=fopen("v_Mg.dat","w");
for i=1:length(r_Mg)
  fprintf(out,"%.15e %.15e\n",r_Mg(i),v_Mg(i));
end
fclose(out);
r_Al=load("r_Al.dat");
v_Al=gszpot(13,r_Al);
out=fopen("v_Al.dat","w");
for i=1:length(r_Al)
  fprintf(out,"%.15e %.15e\n",r_Al(i),v_Al(i));
end
fclose(out);
r_Si=load("r_Si.dat");
v_Si=gszpot(14,r_Si);
out=fopen("v_Si.dat","w");
for i=1:length(r_Si)
  fprintf(out,"%.15e %.15e\n",r_Si(i),v_Si(i));
end
fclose(out);
r_P=load("r_P.dat");
v_P=gszpot(15,r_P);
out=fopen("v_P.dat","w");
for i=1:length(r_P)
  fprintf(out,"%.15e %.15e\n",r_P(i),v_P(i));
end
fclose(out);
r_S=load("r_S.dat");
v_S=gszpot(16,r_S);
out=fopen("v_S.dat","w");
for i=1:length(r_S)
  fprintf(out,"%.15e %.15e\n",r_S(i),v_S(i));
end
fclose(out);
r_Cl=load("r_Cl.dat");
v_Cl=gszpot(17,r_Cl);
out=fopen("v_Cl.dat","w");
for i=1:length(r_Cl)
  fprintf(out,"%.15e %.15e\n",r_Cl(i),v_Cl(i));
end
fclose(out);
r_Ar=load("r_Ar.dat");
v_Ar=gszpot(18,r_Ar);
out=fopen("v_Ar.dat","w");
for i=1:length(r_Ar)
  fprintf(out,"%.15e %.15e\n",r_Ar(i),v_Ar(i));
end
fclose(out);
r_K=load("r_K.dat");
v_K=gszpot(19,r_K);
out=fopen("v_K.dat","w");
for i=1:length(r_K)
  fprintf(out,"%.15e %.15e\n",r_K(i),v_K(i));
end
fclose(out);
r_Ca=load("r_Ca.dat");
v_Ca=gszpot(20,r_Ca);
out=fopen("v_Ca.dat","w");
for i=1:length(r_Ca)
  fprintf(out,"%.15e %.15e\n",r_Ca(i),v_Ca(i));
end
fclose(out);
r_Sc=load("r_Sc.dat");
v_Sc=gszpot(21,r_Sc);
out=fopen("v_Sc.dat","w");
for i=1:length(r_Sc)
  fprintf(out,"%.15e %.15e\n",r_Sc(i),v_Sc(i));
end
fclose(out);
r_Ti=load("r_Ti.dat");
v_Ti=gszpot(22,r_Ti);
out=fopen("v_Ti.dat","w");
for i=1:length(r_Ti)
  fprintf(out,"%.15e %.15e\n",r_Ti(i),v_Ti(i));
end
fclose(out);
r_V=load("r_V.dat");
v_V=gszpot(23,r_V);
out=fopen("v_V.dat","w");
for i=1:length(r_V)
  fprintf(out,"%.15e %.15e\n",r_V(i),v_V(i));
end
fclose(out);
r_Cr=load("r_Cr.dat");
v_Cr=gszpot(24,r_Cr);
out=fopen("v_Cr.dat","w");
for i=1:length(r_Cr)
  fprintf(out,"%.15e %.15e\n",r_Cr(i),v_Cr(i));
end
fclose(out);
r_Mn=load("r_Mn.dat");
v_Mn=gszpot(25,r_Mn);
out=fopen("v_Mn.dat","w");
for i=1:length(r_Mn)
  fprintf(out,"%.15e %.15e\n",r_Mn(i),v_Mn(i));
end
fclose(out);
r_Fe=load("r_Fe.dat");
v_Fe=gszpot(26,r_Fe);
out=fopen("v_Fe.dat","w");
for i=1:length(r_Fe)
  fprintf(out,"%.15e %.15e\n",r_Fe(i),v_Fe(i));
end
fclose(out);
r_Co=load("r_Co.dat");
v_Co=gszpot(27,r_Co);
out=fopen("v_Co.dat","w");
for i=1:length(r_Co)
  fprintf(out,"%.15e %.15e\n",r_Co(i),v_Co(i));
end
fclose(out);
r_Ni=load("r_Ni.dat");
v_Ni=gszpot(28,r_Ni);
out=fopen("v_Ni.dat","w");
for i=1:length(r_Ni)
  fprintf(out,"%.15e %.15e\n",r_Ni(i),v_Ni(i));
end
fclose(out);
r_Cu=load("r_Cu.dat");
v_Cu=gszpot(29,r_Cu);
out=fopen("v_Cu.dat","w");
for i=1:length(r_Cu)
  fprintf(out,"%.15e %.15e\n",r_Cu(i),v_Cu(i));
end
fclose(out);
r_Zn=load("r_Zn.dat");
v_Zn=gszpot(30,r_Zn);
out=fopen("v_Zn.dat","w");
for i=1:length(r_Zn)
  fprintf(out,"%.15e %.15e\n",r_Zn(i),v_Zn(i));
end
fclose(out);
r_Ga=load("r_Ga.dat");
v_Ga=gszpot(31,r_Ga);
out=fopen("v_Ga.dat","w");
for i=1:length(r_Ga)
  fprintf(out,"%.15e %.15e\n",r_Ga(i),v_Ga(i));
end
fclose(out);
r_Ge=load("r_Ge.dat");
v_Ge=gszpot(32,r_Ge);
out=fopen("v_Ge.dat","w");
for i=1:length(r_Ge)
  fprintf(out,"%.15e %.15e\n",r_Ge(i),v_Ge(i));
end
fclose(out);
r_As=load("r_As.dat");
v_As=gszpot(33,r_As);
out=fopen("v_As.dat","w");
for i=1:length(r_As)
  fprintf(out,"%.15e %.15e\n",r_As(i),v_As(i));
end
fclose(out);
r_Se=load("r_Se.dat");
v_Se=gszpot(34,r_Se);
out=fopen("v_Se.dat","w");
for i=1:length(r_Se)
  fprintf(out,"%.15e %.15e\n",r_Se(i),v_Se(i));
end
fclose(out);
r_Br=load("r_Br.dat");
v_Br=gszpot(35,r_Br);
out=fopen("v_Br.dat","w");
for i=1:length(r_Br)
  fprintf(out,"%.15e %.15e\n",r_Br(i),v_Br(i));
end
fclose(out);
r_Kr=load("r_Kr.dat");
v_Kr=gszpot(36,r_Kr);
out=fopen("v_Kr.dat","w");
for i=1:length(r_Kr)
  fprintf(out,"%.15e %.15e\n",r_Kr(i),v_Kr(i));
end
fclose(out);
r_Rb=load("r_Rb.dat");
v_Rb=gszpot(37,r_Rb);
out=fopen("v_Rb.dat","w");
for i=1:length(r_Rb)
  fprintf(out,"%.15e %.15e\n",r_Rb(i),v_Rb(i));
end
fclose(out);
r_Sr=load("r_Sr.dat");
v_Sr=gszpot(38,r_Sr);
out=fopen("v_Sr.dat","w");
for i=1:length(r_Sr)
  fprintf(out,"%.15e %.15e\n",r_Sr(i),v_Sr(i));
end
fclose(out);
r_Y=load("r_Y.dat");
v_Y=gszpot(39,r_Y);
out=fopen("v_Y.dat","w");
for i=1:length(r_Y)
  fprintf(out,"%.15e %.15e\n",r_Y(i),v_Y(i));
end
fclose(out);
r_Zr=load("r_Zr.dat");
v_Zr=gszpot(40,r_Zr);
out=fopen("v_Zr.dat","w");
for i=1:length(r_Zr)
  fprintf(out,"%.15e %.15e\n",r_Zr(i),v_Zr(i));
end
fclose(out);
r_Nb=load("r_Nb.dat");
v_Nb=gszpot(41,r_Nb);
out=fopen("v_Nb.dat","w");
for i=1:length(r_Nb)
  fprintf(out,"%.15e %.15e\n",r_Nb(i),v_Nb(i));
end
fclose(out);
r_Mo=load("r_Mo.dat");
v_Mo=gszpot(42,r_Mo);
out=fopen("v_Mo.dat","w");
for i=1:length(r_Mo)
  fprintf(out,"%.15e %.15e\n",r_Mo(i),v_Mo(i));
end
fclose(out);
r_Tc=load("r_Tc.dat");
v_Tc=gszpot(43,r_Tc);
out=fopen("v_Tc.dat","w");
for i=1:length(r_Tc)
  fprintf(out,"%.15e %.15e\n",r_Tc(i),v_Tc(i));
end
fclose(out);
r_Ru=load("r_Ru.dat");
v_Ru=gszpot(44,r_Ru);
out=fopen("v_Ru.dat","w");
for i=1:length(r_Ru)
  fprintf(out,"%.15e %.15e\n",r_Ru(i),v_Ru(i));
end
fclose(out);
r_Rh=load("r_Rh.dat");
v_Rh=gszpot(45,r_Rh);
out=fopen("v_Rh.dat","w");
for i=1:length(r_Rh)
  fprintf(out,"%.15e %.15e\n",r_Rh(i),v_Rh(i));
end
fclose(out);
r_Pd=load("r_Pd.dat");
v_Pd=gszpot(46,r_Pd);
out=fopen("v_Pd.dat","w");
for i=1:length(r_Pd)
  fprintf(out,"%.15e %.15e\n",r_Pd(i),v_Pd(i));
end
fclose(out);
r_Ag=load("r_Ag.dat");
v_Ag=gszpot(47,r_Ag);
out=fopen("v_Ag.dat","w");
for i=1:length(r_Ag)
  fprintf(out,"%.15e %.15e\n",r_Ag(i),v_Ag(i));
end
fclose(out);
r_Cd=load("r_Cd.dat");
v_Cd=gszpot(48,r_Cd);
out=fopen("v_Cd.dat","w");
for i=1:length(r_Cd)
  fprintf(out,"%.15e %.15e\n",r_Cd(i),v_Cd(i));
end
fclose(out);
r_In=load("r_In.dat");
v_In=gszpot(49,r_In);
out=fopen("v_In.dat","w");
for i=1:length(r_In)
  fprintf(out,"%.15e %.15e\n",r_In(i),v_In(i));
end
fclose(out);
r_Sn=load("r_Sn.dat");
v_Sn=gszpot(50,r_Sn);
out=fopen("v_Sn.dat","w");
for i=1:length(r_Sn)
  fprintf(out,"%.15e %.15e\n",r_Sn(i),v_Sn(i));
end
fclose(out);
r_Sb=load("r_Sb.dat");
v_Sb=gszpot(51,r_Sb);
out=fopen("v_Sb.dat","w");
for i=1:length(r_Sb)
  fprintf(out,"%.15e %.15e\n",r_Sb(i),v_Sb(i));
end
fclose(out);
r_Te=load("r_Te.dat");
v_Te=gszpot(52,r_Te);
out=fopen("v_Te.dat","w");
for i=1:length(r_Te)
  fprintf(out,"%.15e %.15e\n",r_Te(i),v_Te(i));
end
fclose(out);
r_I=load("r_I.dat");
v_I=gszpot(53,r_I);
out=fopen("v_I.dat","w");
for i=1:length(r_I)
  fprintf(out,"%.15e %.15e\n",r_I(i),v_I(i));
end
fclose(out);
r_Xe=load("r_Xe.dat");
v_Xe=gszpot(54,r_Xe);
out=fopen("v_Xe.dat","w");
for i=1:length(r_Xe)
  fprintf(out,"%.15e %.15e\n",r_Xe(i),v_Xe(i));
end
fclose(out);
r_Cs=load("r_Cs.dat");
v_Cs=gszpot(55,r_Cs);
out=fopen("v_Cs.dat","w");
for i=1:length(r_Cs)
  fprintf(out,"%.15e %.15e\n",r_Cs(i),v_Cs(i));
end
fclose(out);
r_Ba=load("r_Ba.dat");
v_Ba=gszpot(56,r_Ba);
out=fopen("v_Ba.dat","w");
for i=1:length(r_Ba)
  fprintf(out,"%.15e %.15e\n",r_Ba(i),v_Ba(i));
end
fclose(out);
r_La=load("r_La.dat");
v_La=gszpot(57,r_La);
out=fopen("v_La.dat","w");
for i=1:length(r_La)
  fprintf(out,"%.15e %.15e\n",r_La(i),v_La(i));
end
fclose(out);
r_Ce=load("r_Ce.dat");
v_Ce=gszpot(58,r_Ce);
out=fopen("v_Ce.dat","w");
for i=1:length(r_Ce)
  fprintf(out,"%.15e %.15e\n",r_Ce(i),v_Ce(i));
end
fclose(out);
r_Pr=load("r_Pr.dat");
v_Pr=gszpot(59,r_Pr);
out=fopen("v_Pr.dat","w");
for i=1:length(r_Pr)
  fprintf(out,"%.15e %.15e\n",r_Pr(i),v_Pr(i));
end
fclose(out);
r_Nd=load("r_Nd.dat");
v_Nd=gszpot(60,r_Nd);
out=fopen("v_Nd.dat","w");
for i=1:length(r_Nd)
  fprintf(out,"%.15e %.15e\n",r_Nd(i),v_Nd(i));
end
fclose(out);
r_Pm=load("r_Pm.dat");
v_Pm=gszpot(61,r_Pm);
out=fopen("v_Pm.dat","w");
for i=1:length(r_Pm)
  fprintf(out,"%.15e %.15e\n",r_Pm(i),v_Pm(i));
end
fclose(out);
r_Sm=load("r_Sm.dat");
v_Sm=gszpot(62,r_Sm);
out=fopen("v_Sm.dat","w");
for i=1:length(r_Sm)
  fprintf(out,"%.15e %.15e\n",r_Sm(i),v_Sm(i));
end
fclose(out);
r_Eu=load("r_Eu.dat");
v_Eu=gszpot(63,r_Eu);
out=fopen("v_Eu.dat","w");
for i=1:length(r_Eu)
  fprintf(out,"%.15e %.15e\n",r_Eu(i),v_Eu(i));
end
fclose(out);
r_Gd=load("r_Gd.dat");
v_Gd=gszpot(64,r_Gd);
out=fopen("v_Gd.dat","w");
for i=1:length(r_Gd)
  fprintf(out,"%.15e %.15e\n",r_Gd(i),v_Gd(i));
end
fclose(out);
r_Tb=load("r_Tb.dat");
v_Tb=gszpot(65,r_Tb);
out=fopen("v_Tb.dat","w");
for i=1:length(r_Tb)
  fprintf(out,"%.15e %.15e\n",r_Tb(i),v_Tb(i));
end
fclose(out);
r_Dy=load("r_Dy.dat");
v_Dy=gszpot(66,r_Dy);
out=fopen("v_Dy.dat","w");
for i=1:length(r_Dy)
  fprintf(out,"%.15e %.15e\n",r_Dy(i),v_Dy(i));
end
fclose(out);
r_Ho=load("r_Ho.dat");
v_Ho=gszpot(67,r_Ho);
out=fopen("v_Ho.dat","w");
for i=1:length(r_Ho)
  fprintf(out,"%.15e %.15e\n",r_Ho(i),v_Ho(i));
end
fclose(out);
r_Er=load("r_Er.dat");
v_Er=gszpot(68,r_Er);
out=fopen("v_Er.dat","w");
for i=1:length(r_Er)
  fprintf(out,"%.15e %.15e\n",r_Er(i),v_Er(i));
end
fclose(out);
r_Tm=load("r_Tm.dat");
v_Tm=gszpot(69,r_Tm);
out=fopen("v_Tm.dat","w");
for i=1:length(r_Tm)
  fprintf(out,"%.15e %.15e\n",r_Tm(i),v_Tm(i));
end
fclose(out);
r_Yb=load("r_Yb.dat");
v_Yb=gszpot(70,r_Yb);
out=fopen("v_Yb.dat","w");
for i=1:length(r_Yb)
  fprintf(out,"%.15e %.15e\n",r_Yb(i),v_Yb(i));
end
fclose(out);
r_Lu=load("r_Lu.dat");
v_Lu=gszpot(71,r_Lu);
out=fopen("v_Lu.dat","w");
for i=1:length(r_Lu)
  fprintf(out,"%.15e %.15e\n",r_Lu(i),v_Lu(i));
end
fclose(out);
r_Hf=load("r_Hf.dat");
v_Hf=gszpot(72,r_Hf);
out=fopen("v_Hf.dat","w");
for i=1:length(r_Hf)
  fprintf(out,"%.15e %.15e\n",r_Hf(i),v_Hf(i));
end
fclose(out);
r_Ta=load("r_Ta.dat");
v_Ta=gszpot(73,r_Ta);
out=fopen("v_Ta.dat","w");
for i=1:length(r_Ta)
  fprintf(out,"%.15e %.15e\n",r_Ta(i),v_Ta(i));
end
fclose(out);
r_W=load("r_W.dat");
v_W=gszpot(74,r_W);
out=fopen("v_W.dat","w");
for i=1:length(r_W)
  fprintf(out,"%.15e %.15e\n",r_W(i),v_W(i));
end
fclose(out);
r_Re=load("r_Re.dat");
v_Re=gszpot(75,r_Re);
out=fopen("v_Re.dat","w");
for i=1:length(r_Re)
  fprintf(out,"%.15e %.15e\n",r_Re(i),v_Re(i));
end
fclose(out);
r_Os=load("r_Os.dat");
v_Os=gszpot(76,r_Os);
out=fopen("v_Os.dat","w");
for i=1:length(r_Os)
  fprintf(out,"%.15e %.15e\n",r_Os(i),v_Os(i));
end
fclose(out);
r_Ir=load("r_Ir.dat");
v_Ir=gszpot(77,r_Ir);
out=fopen("v_Ir.dat","w");
for i=1:length(r_Ir)
  fprintf(out,"%.15e %.15e\n",r_Ir(i),v_Ir(i));
end
fclose(out);
r_Pt=load("r_Pt.dat");
v_Pt=gszpot(78,r_Pt);
out=fopen("v_Pt.dat","w");
for i=1:length(r_Pt)
  fprintf(out,"%.15e %.15e\n",r_Pt(i),v_Pt(i));
end
fclose(out);
r_Au=load("r_Au.dat");
v_Au=gszpot(79,r_Au);
out=fopen("v_Au.dat","w");
for i=1:length(r_Au)
  fprintf(out,"%.15e %.15e\n",r_Au(i),v_Au(i));
end
fclose(out);
r_Hg=load("r_Hg.dat");
v_Hg=gszpot(80,r_Hg);
out=fopen("v_Hg.dat","w");
for i=1:length(r_Hg)
  fprintf(out,"%.15e %.15e\n",r_Hg(i),v_Hg(i));
end
fclose(out);
r_Tl=load("r_Tl.dat");
v_Tl=gszpot(81,r_Tl);
out=fopen("v_Tl.dat","w");
for i=1:length(r_Tl)
  fprintf(out,"%.15e %.15e\n",r_Tl(i),v_Tl(i));
end
fclose(out);
r_Pb=load("r_Pb.dat");
v_Pb=gszpot(82,r_Pb);
out=fopen("v_Pb.dat","w");
for i=1:length(r_Pb)
  fprintf(out,"%.15e %.15e\n",r_Pb(i),v_Pb(i));
end
fclose(out);
r_Bi=load("r_Bi.dat");
v_Bi=gszpot(83,r_Bi);
out=fopen("v_Bi.dat","w");
for i=1:length(r_Bi)
  fprintf(out,"%.15e %.15e\n",r_Bi(i),v_Bi(i));
end
fclose(out);
r_Po=load("r_Po.dat");
v_Po=gszpot(84,r_Po);
out=fopen("v_Po.dat","w");
for i=1:length(r_Po)
  fprintf(out,"%.15e %.15e\n",r_Po(i),v_Po(i));
end
fclose(out);
r_At=load("r_At.dat");
v_At=gszpot(85,r_At);
out=fopen("v_At.dat","w");
for i=1:length(r_At)
  fprintf(out,"%.15e %.15e\n",r_At(i),v_At(i));
end
fclose(out);
r_Rn=load("r_Rn.dat");
v_Rn=gszpot(86,r_Rn);
out=fopen("v_Rn.dat","w");
for i=1:length(r_Rn)
  fprintf(out,"%.15e %.15e\n",r_Rn(i),v_Rn(i));
end
fclose(out);
r_Fr=load("r_Fr.dat");
v_Fr=gszpot(87,r_Fr);
out=fopen("v_Fr.dat","w");
for i=1:length(r_Fr)
  fprintf(out,"%.15e %.15e\n",r_Fr(i),v_Fr(i));
end
fclose(out);
r_Ra=load("r_Ra.dat");
v_Ra=gszpot(88,r_Ra);
out=fopen("v_Ra.dat","w");
for i=1:length(r_Ra)
  fprintf(out,"%.15e %.15e\n",r_Ra(i),v_Ra(i));
end
fclose(out);
r_Ac=load("r_Ac.dat");
v_Ac=gszpot(89,r_Ac);
out=fopen("v_Ac.dat","w");
for i=1:length(r_Ac)
  fprintf(out,"%.15e %.15e\n",r_Ac(i),v_Ac(i));
end
fclose(out);
r_Th=load("r_Th.dat");
v_Th=gszpot(90,r_Th);
out=fopen("v_Th.dat","w");
for i=1:length(r_Th)
  fprintf(out,"%.15e %.15e\n",r_Th(i),v_Th(i));
end
fclose(out);
r_Pa=load("r_Pa.dat");
v_Pa=gszpot(91,r_Pa);
out=fopen("v_Pa.dat","w");
for i=1:length(r_Pa)
  fprintf(out,"%.15e %.15e\n",r_Pa(i),v_Pa(i));
end
fclose(out);
r_U=load("r_U.dat");
v_U=gszpot(92,r_U);
out=fopen("v_U.dat","w");
for i=1:length(r_U)
  fprintf(out,"%.15e %.15e\n",r_U(i),v_U(i));
end
fclose(out);
r_Np=load("r_Np.dat");
v_Np=gszpot(93,r_Np);
out=fopen("v_Np.dat","w");
for i=1:length(r_Np)
  fprintf(out,"%.15e %.15e\n",r_Np(i),v_Np(i));
end
fclose(out);
r_Pu=load("r_Pu.dat");
v_Pu=gszpot(94,r_Pu);
out=fopen("v_Pu.dat","w");
for i=1:length(r_Pu)
  fprintf(out,"%.15e %.15e\n",r_Pu(i),v_Pu(i));
end
fclose(out);
r_Am=load("r_Am.dat");
v_Am=gszpot(95,r_Am);
out=fopen("v_Am.dat","w");
for i=1:length(r_Am)
  fprintf(out,"%.15e %.15e\n",r_Am(i),v_Am(i));
end
fclose(out);
r_Cm=load("r_Cm.dat");
v_Cm=gszpot(96,r_Cm);
out=fopen("v_Cm.dat","w");
for i=1:length(r_Cm)
  fprintf(out,"%.15e %.15e\n",r_Cm(i),v_Cm(i));
end
fclose(out);
r_Bk=load("r_Bk.dat");
v_Bk=gszpot(97,r_Bk);
out=fopen("v_Bk.dat","w");
for i=1:length(r_Bk)
  fprintf(out,"%.15e %.15e\n",r_Bk(i),v_Bk(i));
end
fclose(out);
r_Cf=load("r_Cf.dat");
v_Cf=gszpot(98,r_Cf);
out=fopen("v_Cf.dat","w");
for i=1:length(r_Cf)
  fprintf(out,"%.15e %.15e\n",r_Cf(i),v_Cf(i));
end
fclose(out);
r_Es=load("r_Es.dat");
v_Es=gszpot(99,r_Es);
out=fopen("v_Es.dat","w");
for i=1:length(r_Es)
  fprintf(out,"%.15e %.15e\n",r_Es(i),v_Es(i));
end
fclose(out);
r_Fm=load("r_Fm.dat");
v_Fm=gszpot(100,r_Fm);
out=fopen("v_Fm.dat","w");
for i=1:length(r_Fm)
  fprintf(out,"%.15e %.15e\n",r_Fm(i),v_Fm(i));
end
fclose(out);
r_Md=load("r_Md.dat");
v_Md=gszpot(101,r_Md);
out=fopen("v_Md.dat","w");
for i=1:length(r_Md)
  fprintf(out,"%.15e %.15e\n",r_Md(i),v_Md(i));
end
fclose(out);
r_No=load("r_No.dat");
v_No=gszpot(102,r_No);
out=fopen("v_No.dat","w");
for i=1:length(r_No)
  fprintf(out,"%.15e %.15e\n",r_No(i),v_No(i));
end
fclose(out);
