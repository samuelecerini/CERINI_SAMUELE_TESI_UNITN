import numpy as np
import scipy.constants as constant
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

"""
Simulation Initialization and Configuration Registry.

Defines global physical constants, structural geometries for advanced gravitational 
wave interferometers (ET-LF, ET-HF, Ad-LIGO), laser field parameter 
equations, thermodynamic material datasets, and temporal/spatial discretization 
parameters for the tracking pipeline.
"""


BASE_DIR = Path(__file__).resolve().parent

# COSTANTI FISICHE
SPEED_OF_LIGHT: float = constant.c
EPSILON_0: float      = constant.epsilon_0
MU_0: float           = constant.mu_0
PI: float             = constant.pi
SIGMA_SB: float       = constant.Stefan_Boltzmann  # [W/(m^2 K^4)]
K_B: float            = constant.Boltzmann         # [J/K]
GRAVITY: float        = constant.g                 # [m/s^2]

# PARAMETRI DELLA SIMULAZIONE
# distribuzione con slope
N_PARTICLES: int = 1
REPETITIONS: int = 10
X0_LIMITS: List[float] = [-4900.0, 4900.0]
D_MAX_MICRON: float = 50.0
D_MIN_MICRON: float = 0.5
TM_reduction_length: int = 50                    # 2, 5, 10, 20, 50, 100, 200, 500, 1000

# BOOL DEBUG
SAVE_ALL_PARTICLES_SCATTERED_FIELD: bool = False    
SAVE_FIELDS_AT_TM_FOR_EACH_TIME_INSTANT: bool = False  
SEND_EMAIL_NOTIFICATIONS: bool = False
MAKE_PLOTS: bool = True
DO_FULL_SIMULATION: bool = True
SINGLE_RUN: bool = True

# PARAMETRI SINGLE RUN
SINGLE_PARTICLE_DIAM = 1e-6 * 10               # [m] 
SINGLE_PARTICLE_X0 = 1000                      # [m]
SINGLE_PARTICLE_Y0 = 1e-3 * 0                 # [m]
SINGLE_PARTICLE_REFRACTIVE_INDEX = 1.5 + 0.001j

# Nuovi parametri termodinamici per la particella custom
SINGLE_PARTICLE_RHO = 2650.0                   # Densità [kg/m^3]
SINGLE_PARTICLE_CP = 800.0                     # Calore specifico [J/(kg K)]
SINGLE_PARTICLE_EMISSIVITY = 0.90              # Emissività termica
SINGLE_PARTICLE_L_S = 1.40e7                   # Calore latente di sublimazione [J/kg]
SINGLE_PARTICLE_T_SUB = 2000.0                 # Temperatura di sublimazione [K]


# DISCRETIZZAZIONE TEMPORALE E GRIGLIE DI OSSERVAZIONE
dt: float = 200e-6                             # [s]     time discretization
t_min_obs: float = 0.0                         # [s]     simulation starting time (beginning of interaction with beam)
t_max_obs: float = 10.0                        # [s]     simulation end time (end of interaction with beam)
# una particella ci mette circa 0.5 secondi ad attraversare gli 1.2 metri del tubo
t_min_simulation: float = t_min_obs + 1        # [s]     starting time for particles detachment
t_max_simulation: float = t_max_obs - 1        # [s]     end time for particles detachment


# la simulazione inizia a t_min_obs e finisce a t_max_obs, mentre le particelle si staccano in una finestra contenuta in questo intervallo di tempo (che utilità ha?)
# da implementare: se SINGLE_PARTICLE = True allora non ha senso simulare decine di secondi. basta mettere il tempo di osservazione da 0 al tempo necessario alla particella per percorrere 1.2 metri in caduta, adattandolo alla presenza di eventuali finestre

# CONFIGURAZIONE "ET-HF", "ET-LF", "LIGO"
# NOTA: Questi valori sono i default presi se modifichi il file a mano.
# Verranno sovrascritti dai parser CLI se usi il terminale.
config: str = "ET-HF"  # Valori ammessi: "ET-HF", "ET-LF", "ET-LF-2000nm", "LIGO", "VIRGO"
species: str = "alluminium"  # Valori ammessi: silica, typical, skin, alluminium, inox

# Dichiarazione delle variabili di configurazione (assegnate condizionalmente)
L: float
l: float
w0: float
P0: float
finesse: int
radius_TM: float
M: float
T_env: float
f0: float

tube_radius: float = 0.6  # Raggio del tubo da vuoto [m]

# aggiungere polvere standard
# DATABASE DEI MATERIALI
'''
material_database: Dict[str, Dict[str, Any]] = {
    "silica": {
        "m_1064": 1.450 - 1e-6j,
        "m_1550": 1.444 - 1e-6j,
        "rho": 2200.0,
        "cp": 740.0,
        "emissivity": 0.80,
        "L_s": 1.42e7,
        "T_sub": 2000.0
    },
    "alluminium": {
        "m_1064": 1.25 - 10.5j,
        "m_1550": 1.47 - 16.1j,
        "rho": 2700.0,
        "cp": 900.0,
        "emissivity": 0.10,
        "L_s": 1.08e7,
        "T_sub": 2792.0
    },
    "typical": {                        # Polvere standard / Minerale (es. Arizona Road Dust)
        "m_1064": 1.500 - 1e-3j,        # Indice di rifrazione tipico a 1064 nm[span_1](start_span)[span_1](end_span)
        "m_1550": 1.500 - 1e-3j,        # Indice di rifrazione tipico a 1550 nm
        "rho": 2650.0,                  # Densità media delle polveri minerali [kg/m^3]
        "cp": 800.0,                    # Calore specifico medio [J/(kg K)]
        "emissivity": 0.90,             # Emissività tipica dei silicati
        "L_s": 1.40e7,                  # Calore latente di sublimazione stimato [J/kg]
        "T_sub": 2000.0                 # Temperatura di sublimazione/fusione [K]
    },
    "skin": {
        "m_1064": 1.530 - 5e-4j,
        "m_1550": 1.540 - 1e-3j,        # indice di rifrazione
        "rho": 1100.0,                  # densità [kg/m^3]
        "cp": 1900.0,                   # calore specifico [J/(kg K)]
        "emissivity": 0.95,             # emissività
        "L_s": 2.5e6,                   # calore latente [J/kg]
        "T_sub": 600.0                  # temperatura di sublimazione [K]
    },
    "inox": {                           # Acciaio inossidabile (AISI 304/316)
        "m_1064": 3.10 - 4.40j,       
        "m_1550": 3.45 - 5.32j,      
        "rho": 8000.0,              
        "cp": 500.0,                   
        "emissivity": 0.15,            
        "L_s": 7.4e6,                   
        "T_sub": 3130.0                
    }
}
'''

# DATABASE DEI MATERIALI COMPLETO (23 lug 2026)
material_database = {
    "silica": {
        # Indici di rifrazione complessi (m = n + ik)
        "m_1064": 1.450 - 1e-6j,
        "m_1550": 1.444 - 1e-6j,
        "m_2000": 1.438 - 1e-6j,
        # Proprietà cinematiche e di stato
        "rho": 2200.0,          # Densità [kg/m^3]
        "T_sub": 2000.0,        # Temperatura di sublimazione [K]
        "L_s": 1.42e7,          # Calore latente di sublimazione [J/kg]
        "emissivity": 0.80,     # Emissività termica
        # Calore specifico dinamico [J/(kg K)]
        "cp_290": 740.0,        # Valore a temperatura ambiente (ET-HF / LIGO)
        "cp_10": 2.5            # Valore a regime criogenico (ET-LF) per Debye
    },
    "alluminium": {
        "m_1064": 1.25 - 10.5j,
        "m_1550": 1.47 - 16.1j,
        "m_2000": 2.50 - 21.0j,
        "rho": 2700.0,
        "T_sub": 2792.0,
        "L_s": 1.08e7,
        "emissivity": 0.10,
        # Calore specifico
        "cp_290": 900.0,
        "cp_10": 0.55
    },
    "inox": {
        "m_1064": 3.10 - 4.40j,
        "m_1550": 3.45 - 5.32j,
        "m_2000": 2.18 - 3.89j,
        "rho": 8000.0,
        "T_sub": 3130.0,
        "L_s": 7.4e6,
        "emissivity": 0.15,
        # Calore specifico
        "cp_290": 500.0,
        "cp_10": 1.8
    },
    "typical": {
        "m_1064": 1.500 - 1e-3j,
        "m_1550": 1.500 - 1e-3j,
        "m_2000": 1.480 - 1.5e-3j,
        "rho": 2650.0,
        "T_sub": 2000.0,
        "L_s": 1.40e7,
        "emissivity": 0.90,
        # Calore specifico
        "cp_290": 800.0,
        "cp_10": 3.0
    },
    "skin": {
        "m_1064": 1.530 - 5e-4j,
        "m_1550": 1.540 - 1e-3j,
        "m_2000": 1.380 - 1.0e-3j,
        "rho": 1100.0,
        "T_sub": 600.0,
        "L_s": 2.5e6,
        "emissivity": 0.95,
        # Calore specifico
        "cp_290": 1900.0,
        "cp_10": 4.5
    }
}


# Dichiarazione variabili derivate globali
m: complex
rho_p: float
cp_p: float
emissivity: float
L_s: float
T_sub: float

D_max: float
D_min: float
k: float
nu: float
E0: float
cavity_field_polarization: str = "y"

n_times: int
t_obs: np.ndarray
tau_s: float
fp: float
phi_wp: float = PI / 4.0
em_map_path: Path = BASE_DIR / "ET_TM_maps" / f"map_EM_L0.6_n{TM_reduction_length}.txt"


def update_configuration(target_config: Optional[str] = None, target_species: Optional[str] = None, target_tmgrid: Optional[int] = None) -> None:
    """
    Ricalcola tutte le proprietà fisiche e del campo em in base ai parametri impostati dai parser.
    """
    global config, species, TM_reduction_length
    global L, l, w0, P0, finesse, radius_TM, M, T_env, f0
    global m, rho_p, cp_p, emissivity, L_s, T_sub
    global D_max, D_min, k, nu, E0, n_times, t_obs, tau_s, fp, em_map_path
    
    # sovrascrittura da cli (se forniti argomenti)
    if target_config is not None:
        config = target_config
    if target_species is not None:
        species = target_species
    if target_tmgrid is not None:
        TM_reduction_length = target_tmgrid

    # CONFIGURAZIONI INTERFEROMETRI
    if config == "ET-HF":
        L = 10000.0          # Lunghezza del braccio [m]
        l = 1064e-9          # Lunghezza d'onda del laser [m]
        w0 = 1.42e-2         # Waist del fascio (r0) [m]
        P0 = 3.0e6           # Potenza nel braccio [W]
        finesse = 900        # Finesse della cavità FP
        radius_TM = 0.32     # Raggio della TM [m]
        M = 200.0            # Massa della TM [kg]
        T_env = 290.0        # Temperatura [K]
        f0 = 0.2             # Frequenza caratteristica pendolo [Hz] (SPARATA A CASO)
    elif config == "ET-LF":
        L = 10000.0          
        l = 1550e-9         
        w0 = 2.90e-2        
        P0 = 18.0e3         
        finesse = 900       
        radius_TM = 0.32    
        M = 200.0           
        T_env = 10.0         # Configurazione criogenica
        f0 = 0.2     
    elif config == "ET-LF-2000nm":
        L = 10_000      # arm lenght [m]
        l = 2000e-9     # wavelength [m]
        w0 = 2.90e-2    # waist [m]
        P0 = 18*1e3     # arm power [W]
        finesse = 900   # value from arXiv:2011.02983v1
        radius_TM = 0.32    # [m]
        M = 200         # TM mass (for ET-LF: PhysRevD.108.123009) [kg]    
        T_env = 10.0         # Configurazione criogenica
        f0 = 0.2     
    elif config == "LIGO": 
        L = 4000.0           
        l = 1064e-9         
        w0 = 12e-3          
        P0 = 3.0e5       
        finesse = 450       
        radius_TM = 0.32    
        M = 40.0             
        T_env = 290.0       
        f0 = 0.2
    elif config == "VIRGO":
        L = 3000.0           # arm length [m]
        l = 1064e-9          # wavelength [m]
        w0 = 8e-3            # waist (8mm) [m], from r=6cm on TM (z=1500m)
        P0 = 100e3           # arm power 100_000 W, https://wiki.virgo-gw.eu/Commissioning/OptChar/O3_param
        finesse = 450        # value from https://wiki.virgo-gw.eu/Commissioning/OptChar/WebHome
        radius_TM = 0.32     # [m] (radius for LIGO is 17cm, but we use 32cm for the simulation to avoid issues with the TM map from Jerome)
        M = 40               # TM mass https://dcc-llo.ligo.org/public/0009/G1000098/002/G1000098-v2.pdf [kg]
        T_env = 290.0       
        f0 = 0.2
    else:
        raise ValueError(f"CRITICAL ERROR: Configurazione '{config}' non riconosciuta dal sistema.")

    '''
    if SINGLE_RUN:
        m          = SINGLE_PARTICLE_REFRACTIVE_INDEX
        rho_p      = SINGLE_PARTICLE_RHO
        cp_p       = SINGLE_PARTICLE_CP
        emissivity = SINGLE_PARTICLE_EMISSIVITY
        L_s        = SINGLE_PARTICLE_L_S
        T_sub      = SINGLE_PARTICLE_T_SUB
        print("      ↳ [VARIABLES] Asset termodinamico Single Run Custom applicato con successo.")
    else:
        # standard
        if species not in material_database:
            raise KeyError(f"CRITICAL ERROR: La specie chimica '{species}' non è presente nel database termodinamico.")

        mat_props = material_database[species]
        wl_key = f"m_{int(round(l * 1e9))}"
        m = mat_props[wl_key]
        cp_key = f"cp_{int(round(T_env))}"
        cp_p = mat_props[cp_key]
        rho_p      = mat_props["rho"]
        emissivity = mat_props["emissivity"]
        L_s        = mat_props["L_s"]
        T_sub      = mat_props["T_sub"]
    '''

    if species not in material_database:
        raise KeyError(f"CRITICAL ERROR: La specie chimica '{species}' non è presente nel database termodinamico.")

    mat_props = material_database[species]
    wl_key = f"m_{int(round(l * 1e9))}"
    m = mat_props[wl_key]
    cp_key = f"cp_{int(round(T_env))}"
    cp_p = mat_props[cp_key]
    rho_p      = mat_props["rho"]
    emissivity = mat_props["emissivity"]
    L_s        = mat_props["L_s"]
    T_sub      = mat_props["T_sub"]

    # 4. Calcolo Parametri Derivati (Field Parameters & Cavity)
    D_max = 1e-6 * D_MAX_MICRON
    D_min = 1e-6 * D_MIN_MICRON

    k  = 2.0 * PI / l
    nu = MU_0 * SPEED_OF_LIGHT                       # free space impedence [Z0 ~ 377 Ohm]
    E0 = np.sqrt(4.0 * nu * P0 / (PI * (w0**2)))     # beam amplitude TEM00 [V/m]

    n_times = int(round((t_max_obs - t_min_obs) / dt)) + 1
    t_obs = np.linspace(t_min_obs, t_max_obs, n_times)

    tau_s = (L / SPEED_OF_LIGHT) * finesse / PI   # tempo di accumulo dei fotoni (storage time)
    fp = 1.0 / (4.0 * PI * tau_s)                 # frequenza di taglio del polo di cavità [Hz]

    em_map_path = BASE_DIR / "ET_TM_maps" / f"map_EM_L0.6_n{TM_reduction_length}.txt"


def log_configuration_summary() -> None:
    """
    Prints a structured telemetry summary of the environmental, optical, 
    material, and statistical execution bounds to standard output.
    """
    C1 = 55  # column width

    print("\n" + "="*96)
    print(f"{'EINSTEIN TELESCOPE SIMULATION CONFIGURATION':^96}")
    print("="*96)
    print(f" {f"[STRUMENTAZIONE]  Configurazione: {config}":<{C1}} | Lunghezza Braccio: {L} m")
    print(f" {f"[SISTEMA OTTICO]  Lunghezza d'onda: {l*1e9:.1f} nm":<{C1}} | Finesse: {finesse}")
    print(f" {f"                  Frequenza Polo Cavità: {fp:.3f} Hz":<{C1}} | Storage Time: {tau_s*1e3:.3f} ms")
    print(f" {f"                  Ampiezza di Campo E0: {E0:.3g} V/m":<{C1}} | Potenza P0: {P0/1e3:.1f} kW")
    print(f" {f"[CONTAMINANTE]    Specie Selezionata: {species}":<{C1}} | Indice Rifrazione: {m.real:.3g}{m.imag:+.3g}j")
    print(f" {f"                  Densità (rho): {rho_p} kg/m^3":<{C1}} | T Sublimazione: {T_sub} K")
    print(f" {f"[MONTE CARLO]     Particelle per run: {N_PARTICLES}":<{C1}} | Ripetizioni Totali: {REPETITIONS}")
    print(f" {f"                  Griglia TM: {TM_reduction_length}x{TM_reduction_length}":<{C1}} | Passo temporale: {dt*1e6} us")
    print("="*96 + "\n")


update_configuration()

if not em_map_path.exists():
    raise FileNotFoundError(f"CRITICAL ERROR: Mappa EM mancante per n={TM_reduction_length} nel percorso: {em_map_path}")
