## Compare vanilla LLM to the fine-tuned LLM

import warnings
import numpy as np
import pandas as pd
import mlx.core as mx
import copy
from rouge_score import rouge_scorer
import re
warnings.filterwarnings("ignore", message=".*UNEXPECTED.*") #annoying warnings
warnings.filterwarnings("ignore", message=".*interactive.*")
warnings.filterwarnings("ignore", category=UserWarning)
from transformers import logging
logging.set_verbosity_error()
logging.disable_progress_bar()
logging.set_verbosity(logging.CRITICAL)

from datetime import datetime
from mlx_lm import load, generate
from tqdm import tqdm
from bert_score import score
import json

mx.random.seed(42)

sys_prompt_model1 = """
You are a textbook solution solver.
Use this exact output format: "Reasoning:\n<your reasoning>\n\nSolution:\n<your final answer>".
First provide reasoning, then provide solution in textbook solution-manual style.

You may use the following few shot examples as a guide for how to format your reasoning and solution,
{"prompt": "Using the Helmholtz energy, demonstrate that the pressure for an ideal polyatomic gas is identical to that derived for an ideal monatomic gas in the text. A = -kT ln Q", "completion": "Reasoning:\n No markdown, no extra sections. Just steps.\n\nWe'll outline: start from Helmholtz free energy A = -kT ln Q, where Q is total partition function for N particles (Q = q^N/N!). Then pressure P = -(∂A/∂V)_T. Show derivative leads to NkT/V. Mention that total molecular partition function factorizes into translational, rotational, vibrational, electronic: q = q_T q_R q_V q_E. Only q_T depends on V (q_T = V / λ^3 where λ is thermal de Broglie wavelength L). Then compute derivative of ln q_T with respect to V gives 1/V. Multiply by NkT yields P = NkT/V = nRT/V.\n\nNeed to keep unit logic explicit: k in J/K, T in K, etc.\n\nLet's craft concise reasoning steps.

Start from the Helmholtz free energy for N indistinguishable molecules  \n\n\\[\nA=-k_{\\mathrm B}T\\ln Q,\\qquad \nQ=\\frac{q^{N}}{N!},\n\\]\n\nwhere \\(q\\) is the molecular partition function.  \nThe pressure follows from the thermodynamic definition  \n\n\\[\nP= -\\left(\\frac{\\partial A}{\\partial V}\\right)_{T}\n   = k_{\\mathrm B}T\\left(\\frac{\\partial \\ln Q}{\\partial V}\\right)_{T}.\n\\]\n\nInsert \\(Q\\):\n\n\\[\n\\ln Q = N\\ln q-\\ln N! \\;\\; \\Longrightarrow\\;\\;\n\\frac{\\partial \\ln Q}{\\partial V}=N\\frac{\\partial \\ln q}{\\partial V},\n\\]\n\nbecause the factorial term is volume‑independent. Hence  \n\n\\[\nP = Nk_{\\mathrm B}T\\left(\\frac{\\partial \\ln q}{\\partial V}\\right)_{T}.\n\\]\n\nFactorize the molecular partition function into independent contributions  \n\n\\[\nq = q_T\\,q_R\\,q_V\\,q_E,\n\\]\n\nwith translational (\\(q_T\\)), rotational (\\(q_R\\)), vibrational (\\(q_V\\)) and electronic (\\(q_E\\)) parts.  \nOnly the translational factor depends on volume; the others are functions of temperature (and internal molecular parameters) alone, so  \n\n\\[\n\\frac{\\partial \\ln q}{\\partial V}\n   =\\frac{\\partial }{\\partial V}\\bigl(\\ln q_T+\\ln q_R+\\ln q_V+\\ln q_E\\bigr)\n   =\\frac{\\partial \\ln q_T}{\\partial V}.\n\\]\n\nFor an ideal gas the translational partition function per molecule is  \n\n\\[\nq_T=\\frac{V}{\\Lambda^{3}},\\qquad \n\\Lambda =\\left(\\frac{h^{2}}{2\\pi m k_{\\mathrm B}T}\\right)^{1/2},\n\\]\n\nso \\(q_T\\propto V\\). Therefore  \n\n\\[\n\\ln q_T = \\ln V - 3\\ln\\Lambda,\\qquad \n\\frac{\\partial \\ln q_T}{\\partial V}= \\frac{1}{V}.\n\\]\n\nSubstituting back,\n\n\\[\nP = Nk_{\\mathrm B}T\\,\\frac{1}{V}\n   = \\frac{N k_{\\mathrm B} T}{V}.\n\\]\n\nRecognising \\(n=N/N_A\\) and \\(R=k_{\\mathrm B}N_A\\),\n\n\\[\nP = \\frac{nRT}{V},\n\\]\n\nwhich is the ideal‑gas pressure expression, identical for polyatomic and monatomic gases.\n\nSolution:\nP = - (dA/dV)_T = kT (d ln Q/dV)_T = NkT (d ln q/dV)_T = NkT (d ln (q_T q_R q_V q_E)/dV)_T = NkT (d/dV (ln q_T + ln q_R + ln q_V + ln q_E))_T Recall that only the translational partition function (q_T) is a function of volume, the other partition functions have no volume dependence. Therefore, only the derivative of the term involving q_T is non-vanishing: P = NkT (d ln q_T/dV)_T = NkT (1/q_T) (d q_T/dV)_T = NkT (1/q_T) (d (V/L^3)/dV)_T = NkT/V = nRT/V"}
{"prompt": "A 13.0% solution of K2CO3 by mass has a density of 1.09 g/cm3. Calculate the molality of the solution.", "completion": "Reasoning:\nWe need to outline steps: assume 100 g solution, mass K2CO3 =13 g, water=87 g. Moles K2CO3 =13 g / (138.205 g/mol) ≈0.0941 mol. Molality = moles solute per kg solvent =0.0941 mol /0.087 kg =1.08 mol/kg. Density not needed actually but could be used to confirm volume etc. Provide steps.

Take a convenient sample size of the solution, e.g. 100 g.\n\n**Mass of solute and solvent**  \n- Mass of K₂CO₃ = 13 % × 100 g = 13.0 g  \n- Mass of water (solvent) = 100 g – 13 g = 87.0 g  \n\n**Moles of solute**  \nMolar mass of K₂CO₃ = 2(39.10) + 12.01 + 3(16.00) ≈ 138.20 g mol⁻¹  \n\\[\nn_{\\text{K}_2\\text{CO}_3}= \\frac{13.0\\;\\text{g}}{138.20\\;\\text{g mol}^{-1}}\n= 9.41\\times10^{-2}\\;\\text{mol}\n\\]\n\n**Convert solvent mass to kilograms**  \n\\[\nm_{\\text{solvent}} = 87.0\\;\\text{g}=0.0870\\;\\text{kg}\n\\]\n\n**Molality definition**  \n\\[\nb = \\frac{n_{\\text{solute}}}{m_{\\text{solvent}}}\n= \\frac{9.41\\times10^{-2}\\;\\text{mol}}{0.0870\\;\\text{kg}}\n≈ 1.08\\;\\text{mol kg}^{-1}\n\\]\n\n(The given density (1.09 g cm⁻³) is not required for the molality calculation.)\n\nSolution:\n1.08 mol kg^-1"}
{"prompt": "The singlet state of helium is described by the wave function Ψ = (1/√2)[ψ_1(1)ψ_2(2) + ψ_1(2)ψ_2(1)], where ψ_1 and ψ_2 are the spatial parts of the wave functions. The energy expectation value for this state is given by E_singlet = <Ψ|H|Ψ>. The Hamiltonian H consists of the kinetic energy operators for each electron and the electron-electron repulsion term. The total Hamiltonian H = H_1 + H_2 + V_e-e. The potential energy due to electron-electron repulsion is V_e-e = e^2/(4πε_0|r_1 - r_2|). For the singlet state of helium, the spatial wave functions are ψ_1 = 1s and ψ_2 = 2s. The total energy is calculated as E_singlet = ∫∫ Ψ*(1,2)HΨ(1,2)dr_1dr_2. Letting the operators act on the following wave function gives E_singlet = (1/2)∫∫[1s(1)2s(2) + 2s(1)1s(2)][E_1s(1)2s(2) + E_2s(1)1s(2)]dr_1dr_2 + (1/2)∫∫[1s(1)2s(2) + 2s(1)1s(2)][e^2/(4πε_0|r_1 - r_2|)][1s(1)2s(2) + 2s(1)1s(2)]dr_1dr_2. Expanding the previous expression gives E_singlet = (1/2)E_1s∫∫[1s(1)2s(2) + 2s(1)1s(2)][1s(1)2s(2)]dr_1dr_2 + (1/2)E_2s∫∫[1s(1)2s(2) + 2s(1)1s(2)][2s(1)1s(2)]dr_1dr_2 + (1/2)∫∫[1s(1)2s(2) + 2s(1)1s(2)][e^2/(4πε_0|r_1 - r_2|)][1s(1)2s(2) + 2s(1)1s(2)]dr_1dr_2. Each of the first four integrals of the previous expression can be expanded into the sum of two integrals. For example, (1/2)E_1s∫∫[1s(1)2s(2) + 2s(1)1s(2)][1s(1)2s(2)]dr_1dr_2 = (1/2)E_1s∫∫[1s(1)2s(2)][1s(1)2s(2)]dr_1dr_2 + (1/2)E_1s∫∫[2s(1)1s(2)][1s(1)2s(2)]dr_1dr_2. Because the orbitals are orthogonal and normalized, the first integral has the value one, and the second has the value zero. Therefore, the previous expression can be simplified to E_singlet = E_1s + E_2s + (1/2)∫∫[1s(1)2s(2) + 2s(1)1s(2)][e^2/(4πε_0|r_1 - r_2|)][1s(1)2s(2) + 2s(1)1s(2)]dr_1dr_2. This is the desired result.", "completion": "Reasoning:\n Should outline: define wavefunction, expectation value, expand Hamiltonian terms, use orthonormality, simplify integrals, get result.\n\nProvide equations and logic. No final answer section, just reasoning steps. Ensure explicit equations. No markdown fences.

Start with the singlet spatial wave function for two‑electron He  \n\n\\[\n\\Psi(1,2)=\\frac{1}{\\sqrt 2}\\big[\\psi_{1s}(1)\\psi_{2s}(2)+\\psi_{2s}(1)\\psi_{1s}(2)\\big].\n\\]\n\nThe energy expectation value is  \n\n\\[\nE_{\\text{singlet}}=\\langle \\Psi|H|\\Psi\\rangle\n               =\\int\\!\\!\\int \\Psi^{*}(1,2)\\,H\\,\\Psi(1,2)\\;d\\tau_{1}\\,d\\tau_{2},\n\\]\n\nwith the Hamiltonian  \n\n\\[\nH=H_{1}+H_{2}+V_{e-e},\\qquad \nV_{e-e}= \\frac{e^{2}}{4\\pi\\varepsilon_{0}|{\\bf r}_{1}-{\\bf r}_{2}|}.\n\\]\n\nBecause each one‑electron Hamiltonian \\(H_i\\) acts only on the coordinates of electron \\(i\\),\n\n\\[\nH_{1}\\psi_{a}(1)\\psi_{b}(2)=E_{a}\\,\\psi_{a}(1)\\psi_{b}(2),\\qquad\nH_{2}\\psi_{a}(1)\\psi_{b}(2)=E_{b}\\,\\psi_{a}(1)\\psi_{b}(2),\n\\]\n\nwhere \\(E_{a},E_{b}\\) are the one‑electron orbital energies (here \\(a,b=1s,2s\\)).  \n\nInsert \\(\\Psi\\) and let the operators act:\n\n\\[\n\\begin{aligned}\nE_{\\text{singlet}}&=\\frac12\\!\\int\\!\\!\\int\n   [\\psi_{1s}(1)\\psi_{2s}(2)+\\psi_{2s}(1)\\psi_{1s}(2)]\\\\\n &\\qquad\\times\\big[\\,E_{1s}\\,\\psi_{1s}(1)\\psi_{2s}(2)\n                     +E_{2s}\\,\\psi_{2s}(1)\\psi_{1s}(2) \\\\\n &\\qquad\\quad+ V_{e-e}\\,[\\psi_{1s}(1)\\psi_{2s}(2)+\\psi_{2s}(1)\\psi_{1s}(2)]\\big]\n   \\;d\\tau_{1}d\\tau_{2}.\n\\end{aligned}\n\\]\n\nSeparate the kinetic‑energy contributions from the electron‑repulsion term:\n\n\\[\nE_{\\text{singlet}}=\n\\frac12 E_{1s}\\!\\int\\!\\!\\int[\\psi_{1s}(1)\\psi_{2s}(2)+\\psi_{2s}(1)\\psi_{1s}(2)]\n                               \\,\\psi_{1s}(1)\\psi_{2s}(2)\\,d\\tau_{1}d\\tau_{2}\n+\\frac12 E_{2s}\\!\\int\\!\\!\\int[\\psi_{1s}(1)\\psi_{2s}(2)+\\psi_{2s}(1)\\psi_{1s}(2)]\n                               \\,\\psi_{2s}(1)\\psi_{1s}(2)\\,d\\tau_{1}d\\tau_{2}\n+\\frac12\\!\\int\\!\\!\\int[\\psi_{1s}(1)\\psi_{2s}(2)+\\psi_{2s}(1)\\psi_{1s}(2)]\\\\\n\\qquad\\times V_{e-e}\\,[\\psi_{1s}(1)\\psi_{2s}(2)+\\psi_{2s}(1)\\psi_{1s}(2)]\n   \\;d\\tau_{1}d\\tau_{2}.\n\\]\n\nEach of the first two double integrals expands into two terms.  For example,\n\n\\[\n\\frac12E_{1s}\\!\\int\\!\\!\\int\n   [\\psi_{1s}(1)\\psi_{2s}(2)]\\,\\psi_{1s}(1)\\psi_{2s}(2)\n   +\\frac12E_{1s}\\!\\int\\!\\!\\int\n   [\\psi_{2s}(1)\\psi_{1s}(2)]\\,\\psi_{1s}(1)\\psi_{2s}(2).\n\\]\n\nBecause the orbitals are orthonormal,\n\n\\[\n\\int\n\nSolution:\nE_singlet = E_1s + E_2s + (1/2)∫∫[1s(1)2s(2) + 2s(1)1s(2)][e^2/(4πε_0|r_1 - r_2|)][1s(1)2s(2) + 2s(1)1s(2)]dr_1dr_2. This is the desired result."}
"""

class ModelComparatorSemantics:
    def __init__(self, dataset_dir = "datasets/processed/valid.jsonl", model_type = "allenai/scibert_scivocab_uncased"):
        with open(dataset_dir, "r") as f:
            self.dataset = [json.loads(line) for line in f]
        # new shuffle so that we shuffle the dataset (i.e., if there is inherent order to the textbooks)
        # in reality, this does not matter at all because we run through the entire dataset and compute the average
        # however, it means that the reported metrics would be a proper estimate of the full training if we are like halfway through
        idx = np.random.permutation(len(self.dataset))
        self.dataset = np.array(self.dataset)[idx]
        
        self.model_type = model_type
        self.all_prompts = []
        self.all_ground_truth_completions = []
        self.all_model1_completions = []
        self.all_model2_completions = []

        # BERTScore
        self.bert_precision_model1 = []
        self.bert_precision_model2 = []
        self.bert_recall_model1 = []
        self.bert_recall_model2 = []
        self.bert_f1_model1 = []
        self.bert_f1_model2 = []

        # ROUGE-L
        self.rougeL_precision_model1 = []
        self.rougeL_precision_model2 = []
        self.rougeL_recall_model1 = []
        self.rougeL_recall_model2 = []
        self.rougeL_f1_model1 = []
        self.rougeL_f1_model2 = []

    def _summary_df(self, metrics):
        summary_rows = []
        for metric_name, values in metrics.items():
            arr = np.array(values, dtype=float)
            summary_rows.append(
                {
                    "metric": metric_name,
                    "mean": np.mean(arr) if arr.size else np.nan,
                    "std": np.std(arr) if arr.size else np.nan,
                }
            )
        return pd.DataFrame(summary_rows)

    def compare(self, model_dir1, model_dir2):
        bert_precision_model1 = []
        bert_precision_model2 = []
        bert_recall_model1 = []
        bert_recall_model2 = []
        bert_f1_model1 = []
        bert_f1_model2 = []

        rougeL_precision_model1 = []
        rougeL_precision_model2 = []
        rougeL_recall_model1 = []
        rougeL_recall_model2 = []
        rougeL_f1_model1 = []
        rougeL_f1_model2 = []

        model_1, tokenizer1 = load(model_dir1)
        model_2, tokenizer2 = load(model_dir2)

        for i,set in enumerate(tqdm(self.dataset)):

            messages_1 = [
                {"role": "system", "content": sys_prompt_model1},
                {"role": "user", "content": set['prompt']},
            ]

            messages2 = [
                {"role": "user", "content": set['prompt']},
            ]
            
            
            prompt1 = tokenizer1.apply_chat_template(
                messages_1, add_generation_prompt=True
            )


            prompt2 = tokenizer2.apply_chat_template(
                messages2, add_generation_prompt=True
            )
            

            try:
                from mlx_lm.sample_utils import make_sampler
                text1 = generate(
                    model_1,
                    tokenizer1,
                    prompt=prompt1,
                    verbose=False,
                    max_tokens=5000,
                    sampler=make_sampler(temp=0.6),
                )
                
                text2 = generate(
                    model_2,
                    tokenizer2,
                    prompt=prompt2,
                    verbose=False,
                    max_tokens=5000,
                    sampler=make_sampler(temp=0.6),
                )
                
                orig_text2 = copy.deepcopy(text2)
                
                text2 = text2.split("Solution:\n", 1)[1]

                completion = set['completion'].split("Solution:\n")[1]

                bert_p1, bert_r1, bert_f1_1 = score([text1], [completion], model_type=self.model_type, device='cpu')
                bert_p2, bert_r2, bert_f1_2 = score([text2], [completion], model_type=self.model_type, device='cpu')
                
                bert_precision_model1.append(bert_p1.item())
                bert_precision_model2.append(bert_p2.item())
                bert_recall_model1.append(bert_r1.item())
                bert_recall_model2.append(bert_r2.item())
                bert_f1_model1.append(bert_f1_1.item())
                bert_f1_model2.append(bert_f1_2.item())

                self.all_prompts.append(set['prompt'])
                self.all_ground_truth_completions.append(completion)
                self.all_model1_completions.append(text1)
                self.all_model2_completions.append(text2)


                r_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
                rouge1 = r_scorer.score(completion, text1)['rougeL']
                rouge2 = r_scorer.score(completion, text2)['rougeL']

                rougeL_precision_model1.append(rouge1.precision)
                rougeL_precision_model2.append(rouge2.precision)
                rougeL_recall_model1.append(rouge1.recall)
                rougeL_recall_model2.append(rouge2.recall)
                rougeL_f1_model1.append(rouge1.fmeasure)
                rougeL_f1_model2.append(rouge2.fmeasure)
                
                print(self._summary_df(progress_metrics).to_string())

            except Exception as e:
                print(f"Failed on sample {i}: {e}")
            
            if i % 10 == 0:
                progress_metrics = {
                    "bert_precision_model1": bert_precision_model1,
                    "bert_precision_model2": bert_precision_model2,
                    "bert_recall_model1": bert_recall_model1,
                    "bert_recall_model2": bert_recall_model2,
                    "bert_f1_model1": bert_f1_model1,
                    "bert_f1_model2": bert_f1_model2,
                    "rougeL_precision_model1": rougeL_precision_model1,
                    "rougeL_precision_model2": rougeL_precision_model2,
                    "rougeL_recall_model1": rougeL_recall_model1,
                    "rougeL_recall_model2": rougeL_recall_model2,
                    "rougeL_f1_model1": rougeL_f1_model1,
                    "rougeL_f1_model2": rougeL_f1_model2,
                }
                print(f"\nProgress summary at sample {i}")
                print(self._summary_df(progress_metrics).to_string())


        final_metrics = {
            "bert_precision_model1": bert_precision_model1,
            "bert_precision_model2": bert_precision_model2,
            "bert_recall_model1": bert_recall_model1,
            "bert_recall_model2": bert_recall_model2,
            "bert_f1_model1": bert_f1_model1,
            "bert_f1_model2": bert_f1_model2,
            "rougeL_precision_model1": rougeL_precision_model1,
            "rougeL_precision_model2": rougeL_precision_model2,
            "rougeL_recall_model1": rougeL_recall_model1,
            "rougeL_recall_model2": rougeL_recall_model2,
            "rougeL_f1_model1": rougeL_f1_model1,
            "rougeL_f1_model2": rougeL_f1_model2,
        }

        print("\nFinal summary")
        print(self._summary_df(progress_metrics).to_string())

        self.bert_precision_model1 = bert_precision_model1
        self.bert_precision_model2 = bert_precision_model2
        self.bert_recall_model1 = bert_recall_model1
        self.bert_recall_model2 = bert_recall_model2
        self.bert_f1_model1 = bert_f1_model1
        self.bert_f1_model2 = bert_f1_model2

        self.rougeL_precision_model1 = rougeL_precision_model1
        self.rougeL_precision_model2 = rougeL_precision_model2
        self.rougeL_recall_model1 = rougeL_recall_model1
        self.rougeL_recall_model2 = rougeL_recall_model2
        self.rougeL_f1_model1 = rougeL_f1_model1
        self.rougeL_f1_model2 = rougeL_f1_model2

    def save_results(self):
        if len(self.bert_f1_model1) == 0 or len(self.bert_f1_model2) == 0:
            print("No results to save, please run the compare method first")
            return
        df = pd.DataFrame()
        df['prompt'] = self.all_prompts
        df['ground_truth_completion'] = self.all_ground_truth_completions
        df['model1_completion'] = self.all_model1_completions
        df['model2_completion'] = self.all_model2_completions
        df['bert_precision_model1'] = self.bert_precision_model1
        df['bert_precision_model2'] = self.bert_precision_model2
        df['bert_recall_model1'] = self.bert_recall_model1
        df['bert_recall_model2'] = self.bert_recall_model2
        df['bert_f1_model1'] = self.bert_f1_model1
        df['bert_f1_model2'] = self.bert_f1_model2
        df['rougeL_precision_model1'] = self.rougeL_precision_model1
        df['rougeL_precision_model2'] = self.rougeL_precision_model2
        df['rougeL_recall_model1'] = self.rougeL_recall_model1
        df['rougeL_recall_model2'] = self.rougeL_recall_model2
        df['rougeL_f1_model1'] = self.rougeL_f1_model1
        df['rougeL_f1_model2'] = self.rougeL_f1_model2

        df.to_csv(f"analysis/results/semantics_compare_to_fewshot_pe_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)

        return True
    



if __name__ == "__main__":
    mc = ModelComparatorSemantics(dataset_dir = "/Users/michaelmurray/Documents/GitHub/RPMChem/datasets/current_to_run/valid_IMPUTED.jsonl")
    m1 = "/Users/michaelmurray/.lmstudio/models/personal/8b_noLora"
    m2 = "/Users/michaelmurray/.lmstudio/models/submission/fuse_model_8b_qlora_manual_NEW_prompt"
    mc.compare(m1,m2)
    mc.save_results()

