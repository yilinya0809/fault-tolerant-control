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
register(
    id="NDI-A",
    entry_point="ftc.controllers.NDI.ndi_A:NDIController",
)
register(
    id="NDI-B",
    entry_point="ftc.controllers.NDI.ndi_B:NDIController",
)
register(
    id="NDI-C",
    entry_point="ftc.controllers.NDI.ndi_C:NDIController",
)
register(
    id="INDI",
    entry_point="ftc.controllers.INDI.indi:INDIController",
)
register(
    id="NMPC",
    entry_point="ftc.controllers.MPC.mpc_LC62:MPC",
)
register(
    id="NMPC-DI",
    entry_point="ftc.controllers.MPC.mpc_LC62:NDIController",
)
register(
    id="NMPC-smooth",
    entry_point="ftc.controllers.MPC.smooth:MPC",
)
# register(
#     id="NMPC-smooth-num",
#     entry_point="ftc.controllers.MPC.smooth:Numerical_MPC",
# )
