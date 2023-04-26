from ftc.registration import register

register(
    id="PPC-DSC",
    entry_point="ftc.controllers.PPC-DSC.ppcdsc:DSCController",
)
register(
    id="LQR",
    entry_point="ftc.controllers.LQR.lqr:LQRController",
)
register(
    id="LQR-LC62",
    entry_point="ftc.controllers.LQR.lqr_LC62:LQRController",
)
register(
    id="LQR-LC62-binary",
    entry_point="ftc.controllers.LQR.lqr_LC62_CA:LQR_binaryController",
)
register(
    id="LQR-LC62-FM",
    entry_point="ftc.controllers.LQR.lqr_LC62:LQR_FMController",
)
register(
    id="LQR-LC62-PI",
    entry_point="ftc.controllers.LQR.lqr_LC62_CA:LQR_PIController",
)
register(
    id="LQR-LC62-L1norm",
    entry_point="ftc.controllers.LQR.lqr_LC62_CA:LQR_L1normController",
)
register(
    id="BLF",
    entry_point="ftc.controllers.BLF.BLF_g:BLFController",
)
register(
    id="BLF-LC62",
    entry_point="ftc.controllers.BLF.BLF_g_LC62:BLFController",
)
register(
    id="Adaptive",
    entry_point="ftc.controllers.Adaptive.adaptive:Adaptive",
)
register(
    id="NDI",
    entry_point="ftc.controllers.NDI.ndi:NDIController",
)
