from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    KeepTogether,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "ELEC5305_Project_Proposal_540334953.pdf"

NAVY = colors.HexColor("#17365D")
BLUE = colors.HexColor("#2F75B5")
PALE = colors.HexColor("#EAF2F8")
LIGHT = colors.HexColor("#F5F7FA")
GREY = colors.HexColor("#5B6573")


def register_fonts() -> tuple[str, str]:
    regular = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    bold = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    if regular.exists() and bold.exists():
        pdfmetrics.registerFont(TTFont("ProjectSans", regular))
        pdfmetrics.registerFont(TTFont("ProjectSans-Bold", bold))
        return "ProjectSans", "ProjectSans-Bold"
    return "Helvetica", "Helvetica-Bold"


REGULAR, BOLD = register_fonts()


def header_footer(canvas, doc):
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(colors.HexColor("#D9E2F3"))
    canvas.line(18 * mm, 15 * mm, width - 18 * mm, 15 * mm)
    canvas.setFont(REGULAR, 7.5)
    canvas.setFillColor(GREY)
    canvas.drawString(18 * mm, 10.5 * mm, "ELEC5305 Speech and Audio Processing")
    canvas.drawRightString(width - 18 * mm, 10.5 * mm, f"Revised Project Proposal | {doc.page}")
    canvas.restoreState()


def build() -> None:
    width, height = A4
    doc = BaseDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=20 * mm,
        title="Noise-Augmented Urban Sound Classification under Matched and Mismatched Conditions",
        author="Tongtong Chen",
        subject="ELEC5305 revised project proposal",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="main")
    doc.addPageTemplates(PageTemplate(id="proposal", frames=[frame], onPage=header_footer))

    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "Title",
        parent=styles["Title"],
        fontName=BOLD,
        fontSize=22,
        leading=26,
        textColor=NAVY,
        alignment=TA_LEFT,
        spaceAfter=5 * mm,
    )
    subtitle = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontName=REGULAR,
        fontSize=10.5,
        leading=14,
        textColor=GREY,
        spaceAfter=5 * mm,
    )
    heading = ParagraphStyle(
        "Heading",
        parent=styles["Heading2"],
        fontName=BOLD,
        fontSize=12.5,
        leading=15,
        textColor=NAVY,
        spaceBefore=3.5 * mm,
        spaceAfter=1.5 * mm,
        keepWithNext=True,
    )
    body = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontName=REGULAR,
        fontSize=8.8,
        leading=12.2,
        textColor=colors.HexColor("#20252B"),
        alignment=TA_LEFT,
        spaceAfter=2.2 * mm,
    )
    bullet = ParagraphStyle(
        "Bullet",
        parent=body,
        leftIndent=4 * mm,
        firstLineIndent=-2.5 * mm,
        bulletIndent=0,
        spaceAfter=1.1 * mm,
    )
    small = ParagraphStyle(
        "Small",
        parent=body,
        fontSize=7.4,
        leading=9.7,
        spaceAfter=1.2 * mm,
    )
    table_text = ParagraphStyle(
        "TableText", parent=small, fontSize=7.5, leading=9.2, textColor=colors.HexColor("#222222")
    )
    quote = ParagraphStyle(
        "Question",
        parent=body,
        fontName=BOLD,
        fontSize=10.2,
        leading=14,
        textColor=NAVY,
        leftIndent=5 * mm,
        rightIndent=5 * mm,
        borderColor=BLUE,
        borderWidth=0,
        borderPadding=6,
        backColor=PALE,
        spaceBefore=1.5 * mm,
        spaceAfter=3 * mm,
    )

    story = [
        Paragraph("REVISED PROJECT PROPOSAL", ParagraphStyle("Kicker", parent=small, fontName=BOLD, textColor=BLUE, spaceAfter=2 * mm)),
        Paragraph("Noise-Augmented Urban Sound Classification under Matched and Mismatched Conditions", title),
        Paragraph("Published-baseline reproduction, controlled noise mixing, and interpretable MFCC robustness analysis", subtitle),
    ]

    info = [
        [Paragraph("Student", small), Paragraph("Tongtong Chen", small), Paragraph("Student ID", small), Paragraph("540334953", small)],
        [Paragraph("GitHub", small), Paragraph("tongtongchen-540", small), Paragraph("Revision", small), Paragraph("9 October 2026", small)],
        [Paragraph("Repository", small), Paragraph('<link href="https://github.com/tongtongchen-540/elec5305-project-540334953" color="#2F75B5">github.com/tongtongchen-540/<br/>elec5305-project-540334953</link>', small), Paragraph("Project site", small), Paragraph('<link href="https://tongtongchen-540.github.io/elec5305-project-540334953/" color="#2F75B5">GitHub Pages</link>', small)],
    ]
    info_table = Table(info, colWidths=[22 * mm, 59 * mm, 22 * mm, 59 * mm])
    info_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
        ("TEXTCOLOR", (0, 0), (0, -1), GREY),
        ("TEXTCOLOR", (2, 0), (2, -1), GREY),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D9E2F3")),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("SPAN", (1, 2), (1, 2)),
    ]))
    story.extend([info_table, Spacer(1, 2 * mm), HRFlowable(width="100%", thickness=1.2, color=BLUE)])

    story.extend([
        Paragraph("1. Project Overview", heading),
        Paragraph(
            "Urban sound classifiers recognise events such as sirens, dog barks and engines, but recordings collected in deployment can contain additional traffic, crowds or sensor interference. This project investigates whether training with controlled added noise produces general robustness or merely specialises a classifier to the noise it has already encountered. UrbanSound8K provides the ten target classes and official source-separated folds [1], while selected DEMAND environments provide reproducible real background recordings [5]. The work will retain one fixed MFCC-SVM system so that the added-noise condition is the main experimental variable.",
            body,
        ),
        Paragraph("2. Research Question", heading),
        Paragraph(
            "To what extent does waveform-level noise augmentation improve an MFCC-SVM urban sound classifier when test noise is matched to, or different from, the noise used during training?",
            quote,
        ),
        Paragraph(
            "The contribution is a controlled reproduction-and-extension study: first establish a genuine published baseline, then measure transfer from a known noise to a mismatched or held-out environment, and finally explain the classification changes through signal, spectrum, log-Mel and MFCC analysis.",
            body,
        ),
        Paragraph("3. Background and Literature Rationale", heading),
        Paragraph(
            "Salamon et al. introduced UrbanSound8K and showed that an MFCC baseline was particularly sensitive to background interference and confused acoustically similar classes [1]. MFCCs compactly describe the short-time spectral envelope on a perceptual frequency scale [2], but additive interference changes Mel-band energy and therefore shifts the cepstral representation. Dennis et al. demonstrated that mismatched acoustic conditions can substantially reduce environmental-sound classification performance [3]. Earlier time-frequency work also showed that environmental sounds require features able to represent both transient and sustained behaviour [8].",
            body,
        ),
        Paragraph(
            "Data augmentation has improved environmental-sound models, although gains vary by class and augmentation type [4]. The official UrbanSound8K augmentation resources provide JAMS descriptions and background-noise material generated through the MUDA framework [6]. These resources and established Python libraries will be adapted rather than reimplementing routine dataset and audio operations. Recent CNN systems can learn richer spectro-temporal patterns [4,7], but adding one would confound the present comparison and exceed the intended scope; neural approaches will be discussed but not implemented.",
            body,
        ),
        Paragraph("4. Proposed Methodology", heading),
        Paragraph("4.1 Published baseline reproduction", ParagraphStyle("Sub", parent=heading, fontSize=10.2, leading=12, spaceBefore=1 * mm)),
        Paragraph(
            "The first milestone is to reproduce the 2014 MFCC-RBF-SVM baseline as closely as practical before changing the feature representation. Its stated setup uses approximately 23.2 ms frames with 50% overlap, 40 Mel bands from 0 to 22.05 kHz, the first 25 MFCCs, no pre-emphasis or liftering, clip-level summary statistics, correlation-based feature selection, an RBF SVM and the predefined ten folds [1]. The paper reports 225 dimensions although its description of derivative statistics is ambiguous; the implementation will document the dimension-consistent interpretation and any numerical differences between Essentia/Weka and the available Python libraries. A result reasonably close to the published baseline, rather than an exact matching value, will establish the reference system.",
            body,
        ),
        Paragraph("4.2 Noise mixing and leakage control", ParagraphStyle("Sub2", parent=heading, fontSize=10.2, leading=12, spaceBefore=1 * mm)),
        Paragraph(
            "All UrbanSound8K clips remain in their official folds. The experiment will use synthetic white and pink noise, DEMAND STRAFFIC as traffic noise, and DEMAND PCAFETER as the completely held-out crowd environment [5]. For each clip, signal and zero-mean noise power are computed over the full waveform. Noise is scaled by alpha = sqrt(Ps/(Pn*10^(SNR/10))) and added at 20, 10 or 0 dB. Mixing occurs in floating point; if necessary, one common gain is applied to the complete mixture to prevent clipping without changing SNR. Noise excerpts have the same duration as the target clip and are selected with recorded random seeds. Non-overlapping temporal regions supply training and testing excerpts, with a guard interval between them; PCAFETER is excluded from training entirely.",
            body,
        ),
        Paragraph("4.3 Experimental comparison", ParagraphStyle("Sub3", parent=heading, fontSize=10.2, leading=12, spaceBefore=1 * mm)),
        Paragraph(
            "Four strategies will be compared: original recordings only; original plus white-noise copies; original plus STRAFFIC copies; and original plus a balanced mixture of white, pink and STRAFFIC copies. Each augmented strategy receives the same number of extra clips. Every trained model is tested on original, white, pink, STRAFFIC and PCAFETER conditions at the three SNRs. White-to-white and STRAFFIC-to-STRAFFIC comparisons are matched, other single-noise combinations are mismatched, and PCAFETER measures transfer to an unseen environment. Scaling, feature selection and SVM fitting occur inside each training fold only.",
            body,
        ),
        Paragraph("4.4 Signal-processing and performance analysis", ParagraphStyle("Sub4", parent=heading, fontSize=10.2, leading=12, spaceBefore=1 * mm)),
        Paragraph(
            "Representative waveforms, power spectra, log-Mel spectrograms and MFCC matrices will be shown for every noise type and SNR. Relative Frobenius distance between original and noisy log-Mel/MFCC matrices will quantify representation change, which will then be related to accuracy loss. Classification results will include ten-fold mean and standard deviation for accuracy and macro F1, per-class recall, confusion matrices, accuracy-versus-SNR curves and a training-noise by test-noise heatmap. Runtime, selected feature count and failure cases will also be reported.",
            body,
        ),
        Paragraph("5. Expected Outcomes and Scope", heading),
        Paragraph(
            "The deliverables are reproducible Python code, configuration and run manifests, saved predictions, signal-analysis figures, a written report, audio examples and a brief demonstration video. A single-noise model is expected to benefit most under its matched condition, while mixed-noise training may sacrifice some specialisation but transfer more reliably. A null or negative result remains informative if it shows that simple waveform augmentation does not overcome MFCC sensitivity. The project will not search for the highest UrbanSound8K accuracy or add a neural model; its value lies in a controlled comparison and an interpretable explanation of robustness.",
            body,
        ),
    ])

    timeline_data = [
        [Paragraph("Weeks", table_text), Paragraph("Planned work", table_text), Paragraph("Milestone", table_text)],
        [Paragraph("1-2", table_text), Paragraph("Question, proposal and repository", table_text), Paragraph("Approved proposal", table_text)],
        [Paragraph("3-5", table_text), Paragraph("Literature/code review; dataset audit; exact baseline specification", table_text), Paragraph("Reproduction plan", table_text)],
        [Paragraph("6-7", table_text), Paragraph("Implement and validate the published MFCC-SVM baseline", table_text), Paragraph("Ten-fold baseline", table_text)],
        [Paragraph("8-9", table_text), Paragraph("Noise protocol; waveform, spectrum, log-Mel and MFCC analysis", table_text), Paragraph("Validated mixtures", table_text)],
        [Paragraph("10-11", table_text), Paragraph("Matched/mismatched/held-out experiments and class analysis", table_text), Paragraph("Final measurements", table_text)],
        [Paragraph("12-13", table_text), Paragraph("Report, GitHub documentation, audio examples and video", table_text), Paragraph("Final submission", table_text)],
    ]
    timeline = Table(timeline_data, colWidths=[18 * mm, 104 * mm, 40 * mm], repeatRows=1)
    timeline.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), BOLD),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CCD6E3")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.extend([Paragraph("6. Timeline", heading), timeline])

    references = [
        '[1] J. Salamon, C. Jacoby, and J. P. Bello, “A Dataset and Taxonomy for Urban Sound Research,” <i>ACM Multimedia</i>, pp. 1041-1044, 2014. doi:10.1145/2647868.2655045.',
        '[2] S. B. Davis and P. Mermelstein, “Comparison of Parametric Representations for Monosyllabic Word Recognition in Continuously Spoken Sentences,” <i>IEEE TASSP</i>, 28(4), pp. 357-366, 1980.',
        '[3] J. Dennis, H. D. Tran, and H. Li, “Spectrogram Image Feature for Sound Event Classification in Mismatched Conditions,” <i>IEEE Signal Processing Letters</i>, 18(2), pp. 130-133, 2011.',
        '[4] J. Salamon and J. P. Bello, “Deep Convolutional Neural Networks and Data Augmentation for Environmental Sound Classification,” <i>IEEE Signal Processing Letters</i>, 24(3), pp. 279-283, 2017.',
        '[5] J. Thiemann, N. Ito, and E. Vincent, “The Diverse Environments Multi-channel Acoustic Noise Database (DEMAND),” <i>Proceedings of Meetings on Acoustics</i>, 19, 035081, 2013.',
        '[6] B. McFee et al., “A Software Framework for Musical Data Augmentation,” <i>ISMIR</i>, pp. 248-254, 2015.',
        '[7] K. J. Piczak, “Environmental Sound Classification with Convolutional Neural Networks,” <i>IEEE MLSP</i>, 2015.',
        '[8] S. Chu, S. Narayanan, and C.-C. J. Kuo, “Environmental Sound Recognition with Time-Frequency Audio Features,” <i>IEEE TASLP</i>, 17(6), pp. 1142-1158, 2009.',
    ]
    story.append(Paragraph("7. References", heading))
    story.extend(Paragraph(item, small) for item in references)
    story.extend([
        Spacer(1, 2 * mm),
        KeepTogether([
            HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#D9E2F3")),
            Spacer(1, 1.5 * mm),
            Paragraph(
                "Submission note: The revised PDF, working code and documentation will be maintained in the public repository and project site linked on page 1.",
                ParagraphStyle("Note", parent=small, backColor=PALE, borderPadding=5, textColor=NAVY),
            ),
        ]),
    ])

    doc.build(story)


if __name__ == "__main__":
    build()
