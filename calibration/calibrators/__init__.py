from .loss import *
from .loss_ctc import *
LOSS_FUNC={
    'CE': CrossEntropyLoss,
    'LS': LabelSmoothingLoss,
    'CASR': PairWiseWeightSmoothLoss
}

LOSS_FUNC_CTC={
    'CTC': CTC_Loss,
    'CTCLS': CTCLabelSmoothLoss,
    'CASR': CTCCASLS_v3
}