\ProvidesClass{classes_styles/DiplomaThesisACIN}

\LoadClass[12pt,openright,a4paper,oneside]{scrbook}

\RequirePackage{calc}
\RequirePackage{setspace}
\RequirePackage{graphicx}
\RequirePackage[a4paper,includehead,includefoot]{geometry}
\RequirePackage[Sonny]{fncychap}
\RequirePackage[protrusion=true]{microtype} \microtypecontext{spacing=nonfrench}

% SWITCH: PDFLATEX <-> LATEX
%:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
\newif\ifpdf
\ifx\pdfoutput\undefined
  \pdffalse
\else
  \ifnum\pdfoutput=1
    \pdftrue
  \else
    \pdffalse
  \fi
\fi
%:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
%
%
% DEFINITION OF MARGINS
%:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
\geometry{bindingoffset=0mm}																	% additional offset for book binding
\geometry{top=1.5cm, bottom=1.5cm, left=2.5cm, right=3.0cm}		% margin settings
%
\newlength{\parindentOld}
\setlength{\parindentOld}{\parindent}
%:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
%
%
% PROVIDED COMMANDS
%:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
\providecommand*{\@professor}{<NAME OF PROFESSOR>}
\newcommand{\professor}[1]{\gdef\@professor{#1}} 
%
\providecommand*{\@registrationNbr}{<MATRIKELNUMMER>}
\newcommand{\registrationNbr}[1]{\gdef\@registrationNbr{#1}} 
%
\providecommand*{\@address}{<STRASSE>\\<PLZ> <ORT>, �sterreich}
\newcommand{\address}[1]{\gdef\@address{#1}} 
%
\providecommand*{\@thesismonth}{<MONTH>}
\newcommand{\thesismonth}[1]{\gdef\@thesismonth{#1}} 
%
\providecommand*{\@thesisyear}{<YEAR>}
\newcommand{\thesisyear}[1]{\gdef\@thesisyear{#1}} 
%
\providecommand*{\@acingroup}{<GROUP>}
\newcommand{\acingroup}[1]{\gdef\@acingroup{#1}} 
%:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
%
%
% TITLE PAGE
%:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
\renewcommand{\maketitle}{%
	\newgeometry{nohead,nofoot,textheight=\textheight+\headheight+\footskip} % \topmargin+\headsep
	\setlength{\parindent}{0pt}
	\begin{titlepage}
	\centering
		\begin{spacing}{1.3}
			\begin{figure}[t]
				\ifpdf
					\includegraphics[height=35pt]{../sources/graphics/logos/TUW_color_left.pdf}
					\hfill
					%\includegraphics[height=35pt]{../sources/graphics/logos/ACIN_color_logo_textB.pdf}
				\else
					\includegraphics[height=35pt]{../sources/graphics/logos/TUW_color_left.eps}
					\hfill
					\includegraphics[height=35pt]{../sources/graphics/logos/ACIN_color_logo_textB.eps}
				\fi
			\end{figure}
			\vspace*{20mm}	
			\begin{spacing}{1.7}
				{\Large \textbf{\textsf{\@title}}}\\[7mm]
			    {\Huge \textbf{\textsf{\@subtitle}}}\\[7mm]	
			\end{spacing}
			{\large 384.162 SoC, Advanced Course}\\[81mm]
		\end{spacing}
		\begin{spacing}{1.0}
		    \begin{flushright}
    		    {\large \@author}\\
    			\@registrationNbr\\[10mm]
    				
    			Supervisor:\\
    			{\large \@professor}\\[15mm]
			\end{flushright}
		\end{spacing}		
            

		\vfill
		Wien, im \@thesismonth~\@thesisyear\\[10pt]
		%
	\end{titlepage}
	%
	\restoregeometry
	\setlength{\parindent}{\parindentOld}
}
%:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
