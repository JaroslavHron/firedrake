SetFactory("OpenCASCADE");

A=50.0;
NN=10.0;

h=A;


box = 1;
Box(box)={0,0,0,A,A,A};

cp = newp;
Point(cp) = {0.5*A, 0.5*A, 0.5*A, 0.5*h};
Point{cp} In Volume{box};

s() = Unique(Abs(Boundary{ Volume{box}; }));
l() = Unique(Abs(Boundary{ Surface{s()}; }));
p() = Unique(Abs(Boundary{ Line{l()}; }));
Characteristic Length{p()} = h;
//Transfinite Surface {s()};
//Transfinite Volume { Volume{box} };

Mesh.Algorithm = 6;
Mesh.Algorithm3D = 7;

//Mesh.CharacteristicLengthMin = 1.0*H;
//Mesh.CharacteristicLengthMax = 1.4142*H;

Mesh.CharacteristicLengthExtendFromBoundary = 0;
Mesh.CharacteristicLengthFromPoints = 0;
Mesh.CharacteristicLengthFromCurvature = 0;

Field[1] = MathEval;
Field[1].F = Sprintf("%g", h);
Background Field = 1;


// to match firedrake:
// top = 6, bottom = 5, left and right = 1/2, front and back = 3/4

Physical Volume("cube", 0) = {box};

Physical Surface("top" , 6) = {6};
Physical Surface("front", 3) = {3};
Physical Surface("left", 1) = {1};
Physical Surface("back", 4) = {4};
Physical Surface("bottom", 5) = {5};
Physical Surface("right", 2) = {2};

Mesh.Optimize=1;
Mesh.OptimizeThreshold=0.8;
Mesh.AnisoMax=2.0;
Mesh.Smoothing=100;

Mesh 3;

levels=2;
If ( levels > 0 )
For i In { 1 : levels }
  RefineMesh;
  OptimizeMesh "gmsh";
  EndFor
  EndIf
