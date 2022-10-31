from .loss import CaslsChineseAttnLoss, EntropyRegularAttn, GraduatedLabelSmoothingAttn, PairWiseWeightSmoothLoss, CrossEntropyLoss, LabelSmoothingLoss, LogitMarginL1
from .loss_ctc import CTC_Loss, CTCEntropyRegular, CTCGraduatedLabelSmoothing, CTCLogitMarginL1, CTCLabelSmoothLoss, CTCCASLS_v3, CaslsChineseCTCLoss
LOSS_FUNC={
    'CE':CrossEntropyLoss,
    'LS':LabelSmoothingLoss,
    'MBLS':LogitMarginL1,
    'ER':EntropyRegularAttn,
    'GLS':GraduatedLabelSmoothingAttn,
    'CASLS-EN':PairWiseWeightSmoothLoss,
    'CASLS-ZH':CaslsChineseAttnLoss,
}

LOSS_FUNC_CTC={
    'CTC':CTC_Loss,
    'CTCLS':CTCLabelSmoothLoss,
    'MBLS':CTCLogitMarginL1,
    'ER':CTCEntropyRegular,
    'GLS':CTCGraduatedLabelSmoothing,
    'CASLS-EN':CTCCASLS_v3,
    'CASLS-ZH':CaslsChineseCTCLoss,
}