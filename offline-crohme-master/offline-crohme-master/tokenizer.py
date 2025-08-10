# # import json
# # from tokenizers import Tokenizer
# #
# # # Load the WordPiece tokenizer
# # with open("my_dataset/latex_wordpiece_tokenizer.json", "r", encoding="utf-8") as f:
# #     tokenizer_json = f.read()
# #
# # try:
# #     wordpiece_tokenizer = Tokenizer.from_str(tokenizer_json)
# #     print("✅ WordPiece Tokenizer loaded successfully!\n")
# # except Exception as e:
# #     print("❌ Error loading WordPiece tokenizer:", e)
# #     exit()
# #
# # # Sample LaTeX equations
# # samples = [
# #     r"\frac{a}{b} + \alpha",
# #     r"\sum_{n=1}^{\infty} n^2",
# #     r"\int_0^1 x^2 dx",
# #     r"x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}",
# #     r"\lim_{x \to 0} \frac{\sin x}{x}"
# # ]
# #
# # # Test tokenization
# # for sample in samples:
# #     encoded = wordpiece_tokenizer.encode(sample)
# #     print(f"Equation: {sample}\n")
# #     print("WordPiece Tokenization:", encoded.tokens)
# #     print("-" * 50)
#
# # import re
# # from tokenizers import Tokenizer, models, trainers, pre_tokenizers
# # from tokenizers.pre_tokenizers import PreTokenizer, Split
# # from transformers import PreTrainedTokenizerFast
# #
# # # 1. Define special LaTeX commands
# # special_tokens = [
# #     r"\frac", r"\sin", r"\cos", r"\sqrt", r"\sum",
# #     r"\forall", r"\exists", r"\log", r"\pi", r"\alpha", r"\int", r"\beta", r"\|dots", r"\infty"
# # ]
# #
# # # 2. Build the pattern
# # special_pattern = "|".join(re.escape(token) for token in special_tokens)
# # pattern = rf"({special_pattern}|\$[^\$]+\$|[^\s{{}}_^\\$]+|[{{}}_^\\$])"
# #
# # # 3. Define special tokens with HuggingFace-style tokens
# # special_tokens_dict = {
# #     "pad_token": "[PAD]",
# #     "unk_token": "[UNK]",
# #     "sep_token": "[SEP]",
# #     "cls_token": "[CLS]",
# #     "mask_token": "[MASK]",
# # }
# #
# # # Add LaTeX special tokens
# # for token in special_tokens:
# #     token_name = token.replace("\\", "")
# #     special_tokens_dict[f"latex_{token_name}"] = token
# #
# # # 4. Create initial vocabulary with special tokens
# # initial_vocab = {token: idx for idx, token in enumerate(special_tokens_dict.values())}
# #
# # # 5. Create the base tokenizer with initial vocabulary
# # tokenizer = Tokenizer(models.WordPiece(vocab=initial_vocab, unk_token="[UNK]"))
# #
# # # 6. Set up the pre-tokenizer
# # latex_split = Split(pattern=pattern, behavior='isolated')
# # tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
# #     latex_split,
# #     pre_tokenizers.WhitespaceSplit()
# # ])
# #
# # # 7. Create the trainer
# # trainer = trainers.WordPieceTrainer(
# #     special_tokens=list(special_tokens_dict.values()),
# #     vocab_size=8000,
# #     min_frequency=4
# # )
# #
# #
# # def create_latex_tokenizer(corpus_path):
# #     print(f"Training tokenizer on corpus: {corpus_path}")
# #
# #     # Train the tokenizer on the provided corpus
# #     tokenizer.train([corpus_path], trainer)
# #
# #     # Convert to HuggingFace's PreTrainedTokenizerFast
# #     pretrained_tokenizer = PreTrainedTokenizerFast(
# #         tokenizer_object=tokenizer,
# #         pad_token=special_tokens_dict["pad_token"],
# #         unk_token=special_tokens_dict["unk_token"],
# #         sep_token=special_tokens_dict["sep_token"],
# #         cls_token=special_tokens_dict["cls_token"],
# #         mask_token=special_tokens_dict["mask_token"],
# #     )
# #
# #     # Add special tokens
# #     pretrained_tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
# #
# #     return pretrained_tokenizer
# #
# #
# # if __name__ == "__main__":
# #     # Train tokenizer on your corpus
# #     corpus_path = "corpus.txt"
# #     latex_tokenizer = create_latex_tokenizer(corpus_path)
# #
# #     # Save the trained tokenizer
# #     save_path = "latex_tokenizer"
# #     latex_tokenizer.save_pretrained(save_path)
# #     print(f"Saved trained tokenizer to: {save_path}")
# #
# #     # Test some samples
# #     test_samples = [
# #         r"\frac{a}{b} + \alpha",
# #         r"\sum_{n=1}^{\infty} n^2",
# #         r"\int_0^1 x^2 dx",
# #     ]
# #
# #     print("\nTesting tokenizer on sample expressions:")
# #     for sample in test_samples:
# #         encoded = latex_tokenizer(sample)
# #         decoded = latex_tokenizer.decode(encoded["input_ids"])
# #
# #         print(f"\nOriginal: {sample}")
# #         print(f"Encoded input_ids: {encoded['input_ids']}")
# #         print(f"Decoded: {decoded}")
# #
# #
# # # Save and load functions
# # def save_tokenizer(tokenizer, path):
# #     tokenizer.save_pretrained(path)
# #
# #
# # def load_tokenizer(path):
# #     return PreTrainedTokenizerFast.from_pretrained(path)
#
import re
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from tokenizers.pre_tokenizers import PreTokenizer, Split
from transformers import PreTrainedTokenizerFast

# 1. Define special LaTeX commands
special_tokens = [
    r"\frac", r"\sin", r"\cos", r"\sqrt", r"\sum",
    r"\forall", r"\exists", r"\log", r"\pi", r"\alpha", r"\int", r"\beta", r"\|dots", r"\infty", r"\binom", r"\vec", r"\nabla", r"\to"
]

# 2. Build the pattern
special_pattern = "|".join(re.escape(token) for token in special_tokens)
pattern = rf"({special_pattern}|\$[^\$]+\$|[^\s{{}}_^\\$]+|[{{}}_^\\$])"

# 3. Define special tokens with HuggingFace-style tokens
special_tokens_dict = {
    "pad_token": "[PAD]",
    "unk_token": "[UNK]",
    "sep_token": "[SEP]",
    "cls_token": "[CLS]",
    "mask_token": "[MASK]",
}
for token in special_tokens:
    token_name = token.replace("\\", "")
    special_tokens_dict[f"latex_{token_name}"] = token

# For a Unigram model (SentencePiece–style), we generally train from scratch.
# Optionally, you could pre-seed with special tokens.
# 5. Create the base tokenizer with a Unigram model
tokenizer = Tokenizer(models.Unigram())

# 6. Set up the pre-tokenizer (using your custom regex)
latex_split = Split(pattern=pattern, behavior='isolated')
tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
    latex_split,
    pre_tokenizers.WhitespaceSplit()
])

# 7. Create the trainer using UnigramTrainer
trainer = trainers.UnigramTrainer(
    special_tokens=list(special_tokens_dict.values()),
    vocab_size=8000  # Adjust as needed
)


def create_latex_tokenizer(corpus_path):
    print(f"Training SentencePiece-style (Unigram) tokenizer on corpus: {corpus_path}")
    tokenizer.train([corpus_path], trainer)
    pretrained_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token=special_tokens_dict["pad_token"],
        unk_token=special_tokens_dict["unk_token"],
        sep_token=special_tokens_dict["sep_token"],
        cls_token=special_tokens_dict["cls_token"],
        mask_token=special_tokens_dict["mask_token"],
    )
    pretrained_tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    return pretrained_tokenizer


if __name__ == "__main__":
    corpus_path = "Latex_Corpus_new.txt"  # your corpus file
    latex_tokenizer = create_latex_tokenizer(corpus_path)
    save_path = "latex_tokenizer_sentencepiece"
    latex_tokenizer.save_pretrained(save_path)
    print(f"Saved trained SentencePiece-style tokenizer to: {save_path}")

    # Test samples
    test_samples = [
        r"\frac{a}{b} + \alpha",
        r"\sum_{n=1}^{\infty} n^2",
        r"\int_0^1 x^2 dx",
        r"\lim_{x \to 0} \frac{\sin x}{x}",
        r"\binom{n}{k} = \frac{n!}{k!(n-k)!}",
        r"\sqrt{a^2 + b^2}",
        r"\vec{F} = m\vec{a}",
        r"\nabla \times \vec{F} = 0",
        r"\frac{d^2y}{dx^2} + p(x)\frac{dy}{dx} + q(x)y = g(x)",
    ]
    print("\nTesting SentencePiece-style tokenizer on sample expressions:")
    for sample in test_samples:
        encoded = latex_tokenizer(sample)
        decoded = latex_tokenizer.decode(encoded["input_ids"])
        print(f"\nOriginal: {sample}")
        print(f"Encoded input_ids: {encoded['input_ids']}")
        print(f"Decoded: {decoded}")
#
# import re
# from tokenizers import Tokenizer, models, trainers, pre_tokenizers
# from tokenizers.pre_tokenizers import PreTokenizer, Split
# from transformers import PreTrainedTokenizerFast
#
# # 1. Define special LaTeX commands
# special_tokens = [
#     r"\frac", r"\sin", r"\cos", r"\sqrt", r"\sum",
#     r"\forall", r"\exists", r"\log", r"\pi", r"\alpha", r"\int", r"\beta", r"\|dots", r"\infty"
# ]
#
# # 2. Build the pattern
# special_pattern = "|".join(re.escape(token) for token in special_tokens)
# pattern = rf"({special_pattern}|\$[^\$]+\$|[^\s{{}}_^\\$]+|[{{}}_^\\$])"
#
# # 3. Define special tokens with HuggingFace-style tokens
# special_tokens_dict = {
#     "pad_token": "[PAD]",
#     "unk_token": "[UNK]",
#     "sep_token": "[SEP]",
#     "cls_token": "[CLS]",
#     "mask_token": "[MASK]",
# }
# for token in special_tokens:
#     token_name = token.replace("\\", "")
#     special_tokens_dict[f"latex_{token_name}"] = token
#
# # 4. Create initial vocabulary with special tokens
# initial_vocab = {token: idx for idx, token in enumerate(special_tokens_dict.values())}
#
# # 5. Create the base tokenizer with the BPE model
# # tokenizer = Tokenizer(models.BPE(vocab=initial_vocab, unk_token="[UNK]"))
# tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
#
# # 6. Set up the pre-tokenizer
# latex_split = Split(pattern=pattern, behavior='isolated')
# tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
#     latex_split,
#     pre_tokenizers.WhitespaceSplit()
# ])
#
# # 7. Create the trainer using BpeTrainer
# trainer = trainers.BpeTrainer(
#     special_tokens=list(special_tokens_dict.values()),
#     vocab_size=8000,
#     min_frequency=4
# )
#
#
# def create_latex_tokenizer(corpus_path):
#     print(f"Training BPE tokenizer on corpus: {corpus_path}")
#     tokenizer.train([corpus_path], trainer)
#     pretrained_tokenizer = PreTrainedTokenizerFast(
#         tokenizer_object=tokenizer,
#         pad_token=special_tokens_dict["pad_token"],
#         unk_token=special_tokens_dict["unk_token"],
#         sep_token=special_tokens_dict["sep_token"],
#         cls_token=special_tokens_dict["cls_token"],
#         mask_token=special_tokens_dict["mask_token"],
#     )
#     pretrained_tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
#     return pretrained_tokenizer
#
#
# if __name__ == "__main__":
#     corpus_path = "corpus.txt"  # your corpus file
#     latex_tokenizer = create_latex_tokenizer(corpus_path)
#     save_path = "latex_tokenizer_bpe"
#     latex_tokenizer.save_pretrained(save_path)
#     print(f"Saved trained BPE tokenizer to: {save_path}")
#
#     # Test samples
#     test_samples = [
#         r"\frac{a}{b} + \alpha",
#         r"\sum_{n=1}^{\infty} n^2",
#         r"\int_0^1 x^2 dx",
#         r"\lim_{x \to 0} \frac{\sin x}{x}",
#         r"\binom{n}{k} = \frac{n!}{k!(n-k)!}",
#         r"\sqrt{a^2 + b^2}",
#         r"\vec{F} = m\vec{a}",
#         r"\nabla \times \vec{F} = 0",
#         r"\frac{d^2y}{dx^2} + p(x)\frac{dy}{dx} + q(x)y = g(x)",
#         r"\int_a^b f(x) dx = F(b) - F(a)",
#         r"\sum_{i=0}^n \binom{n}{i} = 2^n",
#         r"\overrightarrow{AB} = \overrightarrow{OB} - \overrightarrow{OA}",
#         r"\frac{\partial f}{\partial x} + \frac{\partial f}{\partial y}",
#         r"f(x) = \frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}",
#         r"\oint_C \vec{F} \cdot d\vec{r} = \iint_S (\nabla \times \vec{F}) \cdot d\vec{S}",
#         r"e^{i\pi} + 1 = 0",
#         r"(a+b)^2 = a^2 + 2ab + b^2",
#         r"\log_a(xy) = \log_a x + \log_a y",
#         r"\sin^2 \theta + \cos^2 \theta = 1",
#         r"\frac{1}{1-x} = \sum_{n=0}^{\infty} x^n \quad \text{for } |x| < 1",
#         r"\triangle ABC \cong \triangle DEF",
#         r"\angle A + \angle B + \angle C = 180^\circ",
#         r"P(A|B) = \frac{P(B|A)P(A)}{P(B)}",
#         r"\det(A) = |A|",
#         r"A \cup B = \{x | x \in A \text{ or } x \in B\}",
#         r"A \cap B = \{x | x \in A \text{ and } x \in B\}",
#         r"\{a_n\}_{n=1}^{\infty}",
#         r"\lfloor x \rfloor \leq x < \lfloor x \rfloor + 1",
#         r"\lceil x \rceil - 1 < x \leq \lceil x \rceil",
#         r"f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}",
#         r"z = re^{i\theta} = r(\cos\theta + i\sin\theta)",
#         r"\forall x \in \mathbb{R}, \exists y \in \mathbb{N}",
#         r"\sum_{k=1}^n k = \frac{n(n+1)}{2}",
#         r"\sum_{k=1}^n k^2 = \frac{n(n+1)(2n+1)}{6}",
#         r"\sum_{k=1}^n k^3 = \left(\frac{n(n+1)}{2}\right)^2",
#         r"\prod_{i=1}^n x_i",
#         r"\int e^x dx = e^x + C",
#         r"\int \sin x \, dx = -\cos x + C",
#         r"\int \cos x \, dx = \sin x + C",
#         r"\int \frac{1}{x} \, dx = \ln|x| + C",
#         r"\int \tan x \, dx = -\ln|\cos x| + C",
#         r"\frac{dx}{dt} = v",
#         r"\frac{d^2x}{dt^2} = a",
#         r"F = G\frac{m_1 m_2}{r^2}",
#         r"E = mc^2",
#         r"PV = nRT",
#         r"F = -k\Delta x",
#         r"W = \int \vec{F} \cdot d\vec{s}",
#         r"P = \frac{dW}{dt}",
#         r"\nabla \cdot \vec{E} = \frac{\rho}{\epsilon_0}",
#         r"\nabla \cdot \vec{B} = 0",
#         r"\nabla \times \vec{E} = -\frac{\partial \vec{B}}{\partial t}",
#         r"\nabla \times \vec{B} = \mu_0\vec{J} + \mu_0\epsilon_0\frac{\partial \vec{E}}{\partial t}",
#         r"i\hbar\frac{\partial\Psi}{\partial t} = -\frac{\hbar^2}{2m}\nabla^2\Psi + V\Psi",
#         r"\Delta x \Delta p \geq \frac{\hbar}{2}",
#         r"S = k_B \ln \Omega",
#         r"A = \pi r^2",
#         r"V = \frac{4}{3}\pi r^3",
#         r"c = \lambda\nu",
#         r"Z = \sum_{i} e^{-\beta E_i}",
#         r"F = -\frac{\partial U}{\partial x}",
#         r"T(n) = aT\left(\frac{n}{b}\right) + f(n)",
#         r"P(X = k) = \binom{n}{k}p^k(1-p)^{n-k}",
#         r"f * g = \int_{-\infty}^{\infty} f(\tau)g(t-\tau)d\tau",
#         r"\mathcal{F}[f(t)] = \int_{-\infty}^{\infty} f(t)e^{-i\omega t}dt",
#         r"\mathcal{L}[f(t)] = \int_{0}^{\infty} f(t)e^{-st}dt",
#         r"\frac{d}{dx}[f(x)g(x)] = f'(x)g(x) + f(x)g'(x)",
#         r"\frac{d}{dx}[f(g(x))] = f'(g(x))g'(x)",
#         r"\int_a^b f(x)dx = \int_a^c f(x)dx + \int_c^b f(x)dx",
#         r"\int_a^b f(x)dx = \int_a^b f(a+b-x)dx",
#         r"\lim_{n\to\infty} \left(1 + \frac{1}{n}\right)^n = e",
#         r"\cos(\alpha \pm \beta) = \cos\alpha\cos\beta \mp \sin\alpha\sin\beta",
#         r"\sin(\alpha \pm \beta) = \sin\alpha\cos\beta \pm \cos\alpha\sin\beta",
#         r"\tan(\alpha \pm \beta) = \frac{\tan\alpha \pm \tan\beta}{1 \mp \tan\alpha\tan\beta}",
#         r"a^n - b^n = (a-b)(a^{n-1} + a^{n-2}b + \cdots + ab^{n-2} + b^{n-1})",
#         r"|z_1 z_2| = |z_1| |z_2|",
#         r"\arg(z_1 z_2) = \arg(z_1) + \arg(z_2)",
#         r"\left|\frac{z_1}{z_2}\right| = \frac{|z_1|}{|z_2|}",
#         r"\arg\left(\frac{z_1}{z_2}\right) = \arg(z_1) - \arg(z_2)",
#         r"e^{i\theta} = \cos\theta + i\sin\theta",
#         r"\sin^2\theta = \frac{1 - \cos(2\theta)}{2}",
#         r"\cos^2\theta = \frac{1 + \cos(2\theta)}{2}",
#         r"\sin\theta\cos\theta = \frac{\sin(2\theta)}{2}",
#         r"\frac{d}{d\theta}[\sin\theta] = \cos\theta",
#         r"\frac{d}{d\theta}[\cos\theta] = -\sin\theta",
#         r"\frac{d}{d\theta}[\tan\theta] = \sec^2\theta",
#         r"\int \sec^2\theta \, d\theta = \tan\theta + C",
#         r"\int \sec\theta\tan\theta \, d\theta = \sec\theta + C",
#         r"\int \csc\theta\cot\theta \, d\theta = -\csc\theta + C",
#         r"\int \csc^2\theta \, d\theta = -\cot\theta + C",
#         r"\oint_{\gamma} f(z) \, dz = 2\pi i \sum_{k=1}^{n} \text{Res}(f, a_k)",
#         r"\oint_{\gamma} \frac{dz}{z-a} = 2\pi i",
#         r"\frac{d^n}{dx^n}[e^{ax}] = a^n e^{ax}",
#         r"\frac{d^n}{dx^n}[\sin(ax)] = a^n \sin\left(ax + \frac{n\pi}{2}\right)",
#         r"\frac{d^n}{dx^n}[\cos(ax)] = a^n \cos\left(ax + \frac{n\pi}{2}\right)",
#         r"\sinh x = \frac{e^x - e^{-x}}{2}",
#         r"\cosh x = \frac{e^x + e^{-x}}{2}",
#         r"\tanh x = \frac{\sinh x}{\cosh x} = \frac{e^x - e^{-x}}{e^x + e^{-x}}",
#         r"\coth x = \frac{\cosh x}{\sinh x} = \frac{e^x + e^{-x}}{e^x - e^{-x}}",
#         r"\text{sech } x = \frac{1}{\cosh x} = \frac{2}{e^x + e^{-x}}",
#         r"\text{csch } x = \frac{1}{\sinh x} = \frac{2}{e^x - e^{-x}}",
#         r"y' + P(x)y = Q(x)",
#         r"y'' + a_1 y' + a_0 y = 0",
#         r"\nabla^2 f = \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2} + \frac{\partial^2 f}{\partial z^2}",
#         r"\int_{\partial\Omega} \omega = \int_{\Omega} d\omega",
#         r"\langle x | y \rangle = \int_{-\infty}^{\infty} x^*(t)y(t)dt",
#         r"\frac{dx}{dt} = f(x,t)",
#         r"x_{n+1} = x_n + hf(x_n, t_n)",
#         r"x_{n+1} = x_n + \frac{h}{2}[f(x_n, t_n) + f(x_n + hf(x_n, t_n), t_{n+1})]",
#         r"\langle \hat{A} \rangle = \langle \psi | \hat{A} | \psi \rangle",
#         r"[\hat{x}, \hat{p}] = i\hbar",
#         r"H\psi = E\psi",
#         r"\frac{d\rho}{dt} + \nabla \cdot (\rho \vec{v}) = 0",
#         r"\frac{\partial \vec{v}}{\partial t} + (\vec{v} \cdot \nabla)\vec{v} = -\frac{1}{\rho}\nabla p + \nu \nabla^2 \vec{v} + \vec{g}",
#         r"\frac{\partial T}{\partial t} + \vec{v} \cdot \nabla T = \alpha \nabla^2 T",
#         r"ds^2 = g_{\mu\nu}dx^{\mu}dx^{\nu}",
#         r"R_{\mu\nu} - \frac{1}{2}Rg_{\mu\nu} + \Lambda g_{\mu\nu} = \frac{8\pi G}{c^4}T_{\mu\nu}",
#         r"\Gamma^{\lambda}_{\mu\nu} = \frac{1}{2}g^{\lambda\sigma}(\partial_{\mu}g_{\nu\sigma} + \partial_{\nu}g_{\mu\sigma} - \partial_{\sigma}g_{\mu\nu})",
#         r"R^{\lambda}_{\mu\nu\kappa} = \partial_{\nu}\Gamma^{\lambda}_{\mu\kappa} - \partial_{\kappa}\Gamma^{\lambda}_{\mu\nu} + \Gamma^{\lambda}_{\sigma\nu}\Gamma^{\sigma}_{\mu\kappa} - \Gamma^{\lambda}_{\sigma\kappa}\Gamma^{\sigma}_{\mu\nu}",
#         r"[A, B] = AB - BA",
#         r"\{A, B\} = AB + BA",
#         r"Tr(AB) = Tr(BA)",
#         r"\det(AB) = \det(A)\det(B)",
#         r"A^{-1}A = AA^{-1} = I",
#         r"\exp(A+B) = \exp(A)\exp(B) \quad \text{if } [A,B]=0",
#         r"(A \otimes B)(C \otimes D) = (AC) \otimes (BD)",
#         r"\langle \psi_1 | \psi_2 \rangle = \int \psi_1^*(x) \psi_2(x) dx",
#         r"P(A \cup B) = P(A) + P(B) - P(A \cap B)",
#         r"P(A \cup B \cup C) = P(A) + P(B) + P(C) - P(A \cap B) - P(A \cap C) - P(B \cap C) + P(A \cap B \cap C)",
#         r"E[X+Y] = E[X] + E[Y]",
#         r"Var(X) = E[X^2] - (E[X])^2",
#         r"Cov(X,Y) = E[XY] - E[X]E[Y]",
#         r"Var(aX + bY) = a^2 Var(X) + b^2 Var(Y) + 2ab\,Cov(X,Y)",
#         r"f_X(x) = \int_{-\infty}^{\infty} f_{X,Y}(x,y) dy",
#         r"F_X(x) = P(X \leq x) = \int_{-\infty}^{x} f_X(t) dt",
#         r"Z = \frac{X - \mu}{\sigma}",
#         r"r = \frac{Cov(X,Y)}{\sqrt{Var(X)Var(Y)}}",
#         r"H(X) = -\sum_{i} p_i \log p_i",
#         r"I(X;Y) = H(X) + H(Y) - H(X,Y)",
#         r"S(X,Y) = k_B \ln \Omega(X,Y)",
#         r"\frac{\partial S}{\partial E} = \frac{1}{T}",
#         r"dE = TdS - PdV + \mu dN",
#         r"F = E - TS",
#         r"G = F + PV = E - TS + PV",
#         r"H = E + PV",
#         r"dG = -SdT + VdP + \mu dN",
#         r"\frac{\partial G}{\partial P} = V",
#         r"C_V = \left(\frac{\partial E}{\partial T}\right)_V",
#         r"C_P = \left(\frac{\partial H}{\partial T}\right)_P",
#         r"\gamma = \frac{C_P}{C_V}",
#         r"e^{ax}\sin(bx) = \frac{e^{ax}}{2i}(e^{ibx} - e^{-ibx})",
#         r"e^{ax}\cos(bx) = \frac{e^{ax}}{2}(e^{ibx} + e^{-ibx})",
#         r"\sin(nx) = \sum_{k=0}^{\lfloor\frac{n-1}{2}\rfloor} (-1)^k \binom{n}{2k+1} \cos^{n-2k-1}(x) \sin^{2k+1}(x)",
#         r"\cos(nx) = \sum_{k=0}^{\lfloor\frac{n}{2}\rfloor} (-1)^k \binom{n}{2k} \cos^{n-2k}(x) \sin^{2k}(x)",
#         r"\frac{d}{dx}[\ln(f(x))] = \frac{f'(x)}{f(x)}",
#         r"\int \frac{f'(x)}{f(x)} dx = \ln|f(x)| + C",
#         r"\sum_{k=0}^{\infty} x^k = \frac{1}{1-x}, \quad |x| < 1",
#         r"\sum_{k=0}^{\infty} kx^{k-1} = \frac{1}{(1-x)^2}, \quad |x| < 1",
#         r"\sum_{k=0}^{\infty} \frac{x^k}{k!} = e^x",
#         r"\sum_{k=0}^{\infty} \frac{(-1)^k x^{2k+1}}{(2k+1)!} = \sin x",
#         r"\sum_{k=0}^{\infty} \frac{(-1)^k x^{2k}}{(2k)!} = \cos x",
#         r"f(x) = \frac{a_0}{2} + \sum_{n=1}^{\infty} \left[ a_n\cos\left(\frac{n\pi x}{L}\right) + b_n\sin\left(\frac{n\pi x}{L}\right) \right]",
#         r"a_n = \frac{1}{L}\int_{-L}^{L} f(x)\cos\left(\frac{n\pi x}{L}\right)dx",
#         r"b_n = \frac{1}{L}\int_{-L}^{L} f(x)\sin\left(\frac{n\pi x}{L}\right)dx",
#         r"f(x) = \frac{1}{2\pi}\int_{-\infty}^{\infty} \hat{f}(\omega)e^{i\omega x}d\omega",
#         r"\hat{f}(\omega) = \int_{-\infty}^{\infty} f(x)e^{-i\omega x}dx",
#         r"\frac{d^n}{dx^n}\left[f(x)g(x)\right] = \sum_{k=0}^{n} \binom{n}{k} f^{(k)}(x)g^{(n-k)}(x)",
#         r"T_n(x) = \cos(n\arccos x)",
#         r"U_n(x) = \frac{\sin((n+1)\arccos x)}{\sin(\arccos x)}",
#         r"L_n(x) = \frac{e^x}{n!}\frac{d^n}{dx^n}(x^ne^{-x})",
#         r"H_n(x) = (-1)^n e^{x^2} \frac{d^n}{dx^n}(e^{-x^2})",
#         r"P_n(x) = \frac{1}{2^n n!}\frac{d^n}{dx^n}[(x^2-1)^n]",
#         r"C_n^{(\alpha)}(x) = \frac{1}{n!}\left(\frac{d}{dx}\right)^n (1-x^2)^{n+\alpha-\frac{1}{2}}",
#         r"\Lambda = -g^{\mu\nu}\Gamma^{\lambda}_{\mu\nu}\partial_{\lambda}",
#         r"\Box\phi = \frac{1}{\sqrt{-g}}\partial_{\mu}(\sqrt{-g}g^{\mu\nu}\partial_{\nu}\phi)",
#         r"F_{\mu\nu} = \partial_{\mu}A_{\nu} - \partial_{\nu}A_{\mu}",
#         r"\mathcal{L} = -\frac{1}{4}F_{\mu\nu}F^{\mu\nu} - J^{\mu}A_{\mu}",
#         r"\hat{\psi}(x) = \sum_s \int \frac{d^3p}{(2\pi)^3} \frac{1}{\sqrt{2E_{\vec{p}}}} [a_{\vec{p},s} u_s(\vec{p})e^{-ip\cdot x} + b_{\vec{p},s}^{\dagger}v_s(\vec{p})e^{ip\cdot x}]",
#         r"\frac{1}{4!}\phi^4",
#         r"\bar{\psi}(i\gamma^{\mu}\partial_{\mu} - m)\psi",
#         r"G(x,y) = \langle 0 | T\{\phi(x)\phi(y)\} | 0 \rangle",
#         r"\langle \Omega | T\{e^{i\int d^4x \mathcal{L}_{int}(x)}\} | \Omega \rangle",
#         r"\delta(x-y) = \sum_n \psi_n^*(y)\psi_n(x)",
#         r"\langle a | \hat{T} | b \rangle = \delta_{ab}",
#         r"\sigma_x = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}",
#         r"\sigma_y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}",
#         r"\sigma_z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}",
#         r"[\sigma_i, \sigma_j] = 2i\epsilon_{ijk}\sigma_k",
#         r"\{ \gamma^{\mu}, \gamma^{\nu} \} = 2g^{\mu\nu}",
#         r"\psi(r,\theta,\phi) = R_{nl}(r)Y_{lm}(\theta,\phi)",
#         r"Y_{lm}(\theta,\phi) = \sqrt{\frac{(2l+1)(l-m)!}{4\pi(l+m)!}}P_l^m(\cos\theta)e^{im\phi}",
#         r"R_{nl}(r) = \sqrt{\left(\frac{2}{na_0}\right)^3\frac{(n-l-1)!}{2n[(n+l)!]^3}}\left(\frac{2r}{na_0}\right)^l e^{-r/na_0} L_{n-l-1}^{2l+1}\left(\frac{2r}{na_0}\right)",
#         r"|\psi\rangle = \alpha|0\rangle + \beta|1\rangle",
#         r"|\Psi\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)",
#         r"\rho = \sum_i p_i |\psi_i\rangle\langle\psi_i|",
#         r"S(\rho) = -\text{Tr}(\rho\ln\rho)",
#         r"Q(\rho) = 1 - \langle\psi|\rho|\psi\rangle",
#         r"F(\rho, \sigma) = \text{Tr}\sqrt{\sqrt{\rho}\sigma\sqrt{\rho}}",
#         r"C(\rho) = \max\{0, \lambda_1 - \lambda_2 - \lambda_3 - \lambda_4\}",
#         r"E(\rho) = -\text{Tr}(\rho_A \log \rho_A)",
#         r"D(\rho||\sigma) = \text{Tr}(\rho\log\rho - \rho\log\sigma)",
#         r"M = \sum_k E_k \rho E_k^{\dagger}",
#         r"\Lambda(\rho) = \sum_k A_k \rho A_k^{\dagger}",
#         r"c^{(i)} = \sum_{\{j\}} M^{(i,j)} c^{(j)}",
#         r"|G\rangle = \frac{1}{\sqrt{|G|}}\sum_{g \in G} |g\rangle",
#         r"A \otimes B = \begin{pmatrix} a_{11}B & \cdots & a_{1n}B \\ \vdots & \ddots & \vdots \\ a_{m1}B & \cdots & a_{mn}B \end{pmatrix}",
#         r"|(i,j)\rangle = |i\rangle \otimes |j\rangle",
#         r"U|\psi\rangle = \sum_j c_j |j\rangle",
#         r"R_x(\theta) = e^{-i\theta X/2} = \cos(\theta/2)I - i\sin(\theta/2)X",
#         r"R_y(\theta) = e^{-i\theta Y/2} = \cos(\theta/2)I - i\sin(\theta/2)Y",
#         r"R_z(\theta) = e^{-i\theta Z/2} = \cos(\theta/2)I - i\sin(\theta/2)Z",
#         r"X = |0\rangle\langle 1| + |1\rangle\langle 0|",
#         r"Y = -i|0\rangle\langle 1| + i|1\rangle\langle 0|",
#         r"Z = |0\rangle\langle 0| - |1\rangle\langle 1|",
#         r"H = \frac{1}{\sqrt{2}}(X + Z)",
#         r"CNOT = |0\rangle\langle 0| \otimes I + |1\rangle\langle 1| \otimes X",
#         r"\text{SWAP} = |00\rangle\langle 00| + |01\rangle\langle 10| + |10\rangle\langle 01| + |11\rangle\langle 11|",
#         r"T = \begin{pmatrix} 1 & 0 \\ 0 & e^{i\pi/4} \end{pmatrix}",
#         r"S = \begin{pmatrix} 1 & 0 \\ 0 & i \end{pmatrix}",
#         r"H = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}",
#     ]
#     print("\nTesting BPE tokenizer on sample expressions:")
#     for sample in test_samples:
#         encoded = latex_tokenizer(sample)
#         decoded = latex_tokenizer.decode(encoded["input_ids"])
#         print(f"\nOriginal: {sample}")
#         print(f"Encoded input_ids: {encoded['input_ids']}")
#         print(f"Decoded: {decoded}")
