# Code excerpts (current working tree)

BanhmiTTS HEAD `0a9bec22e314023e7c3a01f52fc8844f9e323e2c`; tracked changes under banhmi_train/configs: `M banhmi_train/export_onnx.py`.
Piper HEAD `73c04d81d5590ecc46e522de3601ce7fb29fc2be`.
Current source is not proof of historical training-time source (see Q03 §11).

## BanhmiTTS:banhmi_train/vits/modules/generator.py
SHA-256 `d92fb3703b21827d0a320f70e30b23af594003fe7092acca5f1315cb7b4ac475`, 180 lines

### lines 18–24
```python
  18  # The inverted-residual resblocks were designed and trained as one bundle
  19  # with SnakeBeta activations and the MRD discriminator (see VitsModel); the
  20  # HiFi-GAN resblocks use LeakyReLU and no MRD. Every released checkpoint
  21  # follows this pairing, so it is fixed here rather than exposed as flags.
  22  INVERTED_RESBLOCKS = ("mb", "mrf")
  23  
  24  
```

### lines 25–60
```python
  25  class Generator(nn.Module):
  26      def __init__(
  27          self,
  28          initial_channel: int,
  29          resblock: str,
  30          resblock_kernel_sizes: typing.Tuple[int, ...],
  31          resblock_dilation_sizes: typing.Tuple[typing.Tuple[int, ...], ...],
  32          upsample_rates: typing.Tuple[int, ...],
  33          upsample_initial_channel: int,
  34          upsample_kernel_sizes: typing.Tuple[int, ...],
  35          gin_channels: int = 0,
  36          mb_expansion: typing.Union[int, typing.Tuple[int, ...]] = 6,
  37          mb_kernel_size: int = 3,
  38          mb_blocks_per_stage: int = 2,
  39          mb_mrf_kernel_sizes: typing.Tuple[int, ...] = (3, 5, 7),
  40      ):
  41          super().__init__()
  42          # "mb" -- new novelty direction (MobileNetV2-style inverted-residual
  43          # blocks, see ResBlockInverted's docstring), validated by overfit-test
  44          # sweep in test_function/compare_generators.py before wiring here:
  45          # expansion=6/kernel=3/blocks_per_stage=2 (true MobileNetV2 defaults)
  46          # beat both resblock "1" and "2" at a matched parameter budget, and
  47          # zero-initializing ResBlockInverted's last pointwise conv (so each
  48          # block starts as an identity no-op) was what actually fixed training
  49          # instability -- not a smaller learning rate, which only masked it.
  50          # This path is fully separate from "1"/"2" below (own attribute,
  51          # own forward()/remove_weight_norm() branch) so existing "1"/"2"
  52          # checkpoints' state_dict keys (`resblocks.*`) are untouched.
  53          self.resblock_mode = resblock
  54          self.num_kernels = len(resblock_kernel_sizes)
  55          self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
  56          activation_cls = SnakeBeta if resblock in INVERTED_RESBLOCKS else LeakyReLUActivation
  57  
  58          self.ups = nn.ModuleList()
  59          self.pre_up_snakes = nn.ModuleList()
  60          for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
```

### lines 61–73
```python
  61              self.pre_up_snakes.append(activation_cls(upsample_initial_channel // (2**i)))
  62              self.ups.append(
  63                  weight_norm(
  64                      ConvTranspose1d(
  65                          upsample_initial_channel // (2**i),
  66                          upsample_initial_channel // (2 ** (i + 1)),
  67                          k,
  68                          u,
  69                          padding=(k - u) // 2,
  70                      )
  71                  )
  72              )
  73  
```

### lines 74–143
```python
  74          if resblock == "mb":
  75              # mb_expansion can be a single int (same expansion at every
  76              # stage, the original config) or a per-stage tuple -- added so
  77              # the last stage (T=8192, the raw-sample-rate stage) can use a
  78              # smaller expansion than the earlier stages (T=256/2048): a
  79              # profiling pass (see docs -- RTF was ~3.7x baseline at uniform
  80              # expansion=6) found the expand/project pointwise convs' cost
  81              # scales with expansion AND with T, so the same expansion is far
  82              # more expensive at the last stage than the first. Earlier
  83              # stages' T is small enough that keeping expansion=6 there stays
  84              # cheap even though it's the same ratio.
  85              if isinstance(mb_expansion, int):
  86                  mb_expansion = (mb_expansion,) * len(self.ups)
  87              assert len(mb_expansion) == len(self.ups), (
  88                  f"mb_expansion must be an int or a tuple with one entry per upsample stage "
  89                  f"({len(self.ups)}), got {mb_expansion}"
  90              )
  91              self.resblocks = None
  92              self.mb_blocks = nn.ModuleList()
  93              for i in range(len(self.ups)):
  94                  ch = upsample_initial_channel // (2 ** (i + 1))
  95                  self.mb_blocks.append(
  96                      nn.ModuleList(
  97                          ResBlockInverted(ch, kernel_size=mb_kernel_size, activation_cls=activation_cls, expansion=mb_expansion[i])
  98                          for _ in range(mb_blocks_per_stage)
  99                      )
 100                  )
 101          elif resblock == "mrf":
 102              # HiFi-GAN's real MRF mechanism (N branches run in PARALLEL on
 103              # the same input, summed then divided by N -- exactly what the
 104              # "else" branch below does for resblock "1"/"2") but each
 105              # branch is a ResBlockInverted instead of ResBlock1/2, one
 106              # branch per entry in mb_mrf_kernel_sizes. Validated by
 107              # overfit-test sweep (test_function/compare_generators.py's
 108              # `_HiFiGANInvertedMRF`) against "mb" sequential-stacking above,
 109              # across 4 per-stage expansion schedules (docs/
 110              # mbconv_resblock_report.md SS6): at expansion=(1,1,1) this is
 111              # the only MRF schedule where sequential doesn't clearly win on
 112              # both loss_mel and speed -- MRF(1,1,1) trades a small amount
 113              # of speed (still faster than plain resblock "2" in isolated
 114              # generator benchmarks, though those were noisy -- see report)
 115              # against sequential(1,1,1) for a chance at the multi-receptive-
 116              # field mechanism HiFi-GAN's own design actually uses.
 117              if isinstance(mb_expansion, int):
 118                  mb_expansion = (mb_expansion,) * len(self.ups)
 119              assert len(mb_expansion) == len(self.ups), (
 120                  f"mb_expansion must be an int or a tuple with one entry per upsample stage "
 121                  f"({len(self.ups)}), got {mb_expansion}"
 122              )
 123              self.resblocks = None
 124              self.mb_blocks = nn.ModuleList()
 125              for i in range(len(self.ups)):
 126                  ch = upsample_initial_channel // (2 ** (i + 1))
 127                  self.mb_blocks.append(
 128                      nn.ModuleList(
 129                          ResBlockInverted(ch, kernel_size=ks, activation_cls=activation_cls, expansion=mb_expansion[i])
 130                          for ks in mb_mrf_kernel_sizes
 131                      )
 132                  )
 133          else:
 134              resblock_cls = ResBlock1 if resblock == "1" else ResBlock2
 135              self.mb_blocks = None
 136              self.resblocks = nn.ModuleList()
 137              for i in range(len(self.ups)):
 138                  ch = upsample_initial_channel // (2 ** (i + 1))
 139                  for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
 140                      self.resblocks.append(resblock_cls(ch, k, d, activation_cls=activation_cls))
 141  
 142          self.final_snake = activation_cls(ch)
 143          self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
```

### lines 145–169
```python
 145  
 146          if gin_channels != 0:
 147              self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)
 148  
 149      def forward(self, x: torch.Tensor, g=None) -> torch.Tensor:
 150          x = self.conv_pre(x)
 151          if g is not None:
 152              x = x + self.cond(g)
 153  
 154          for i, up in enumerate(self.ups):
 155              x = up(self.pre_up_snakes[i](x))
 156              if self.resblock_mode == "mb":
 157                  for block in self.mb_blocks[i]:
 158                      x = block(x)
 159              elif self.resblock_mode == "mrf":
 160                  branches = self.mb_blocks[i]
 161                  x = sum(branch(x) for branch in branches) / len(branches)
 162              else:
 163                  xs = sum(
 164                      self.resblocks[i * self.num_kernels + j](x) for j in range(self.num_kernels)
 165                  )
 166                  x = xs / self.num_kernels
 167  
 168          x = self.conv_post(self.final_snake(x))
 169          return torch.tanh(x)
```

## BanhmiTTS:banhmi_train/vits/utils/resblocks.py
SHA-256 `6c5c9422529a056a733a84d47739076ecb70dbbee3e84a6f6a9ee3a93444efe5`, 172 lines

### lines 70–106
```python
  70  class ResBlock2(nn.Module):
  71      """Two-conv-pair variant of ResBlock1 (fewer dilations, lighter)."""
  72  
  73      def __init__(
  74          self,
  75          channels: int,
  76          kernel_size: int = 3,
  77          dilation: typing.Tuple[int, ...] = (1, 3),
  78          activation_cls: ActivationCls = SnakeBeta,
  79      ):
  80          super().__init__()
  81          self.snakes = nn.ModuleList([activation_cls(channels) for _ in dilation])
  82          self.convs = nn.ModuleList(
  83              [
  84                  weight_norm(
  85                      Conv1d(channels, channels, kernel_size, 1, dilation=d, padding=get_padding(kernel_size, d))
  86                  )
  87                  for d in dilation
  88              ]
  89          )
  90          self.convs.apply(init_weights)
  91  
  92      def forward(self, x: torch.Tensor, x_mask=None) -> torch.Tensor:
  93          for snake, conv in zip(self.snakes, self.convs):
  94              xt = snake(x)
  95              if x_mask is not None:
  96                  xt = xt * x_mask
  97              x = conv(xt) + x
  98          if x_mask is not None:
  99              x = x * x_mask
 100          return x
 101  
 102      def remove_weight_norm(self):
 103          for layer in self.convs:
 104              remove_weight_norm(layer)
 105  
 106  
```

### lines 107–172
```python
 107  class ResBlockInverted(nn.Module):
 108      """Inverted-residual / MBConv-style block (MobileNetV2, Sandler et al.
 109      2018 -- also the shape of Conformer's convolution module): pointwise
 110      (expand) -> depthwise -> pointwise (project), single residual around
 111      the whole block, replacing ResBlock1/ResBlock2's parallel dilated-conv
 112      branches entirely (one block per stage, not multiple summed branches).
 113  
 114      Two deliberate departures from the two blocks above, both novel to
 115      this direction (not MarGan): weight_norm instead of MobileNetV2's
 116      BatchNorm (matches every other conv in this Generator, and avoids
 117      BatchNorm's batch-size sensitivity during GAN training), and the
 118      project's own SnakeBeta instead of ReLU6.
 119  
 120      Linear bottleneck (MobileNetV2's own term): the last pointwise conv
 121      has NO activation after it -- the paper's finding that nonlinearities
 122      destroy information when applied to an already-projected-down (here:
 123      back to `channels`, not the expanded `channels*expansion`) representation.
 124      """
 125  
 126      def __init__(
 127          self,
 128          channels: int,
 129          kernel_size: int = 7,
 130          dilation: int = 1,
 131          activation_cls: ActivationCls = SnakeBeta,
 132          expansion: int = 2,
 133      ):
 134          super().__init__()
 135          hidden = channels * expansion
 136          self.pw_expand = weight_norm(Conv1d(channels, hidden, 1))
 137          self.act1 = activation_cls(hidden)
 138          self.dw = weight_norm(
 139              Conv1d(hidden, hidden, kernel_size, dilation=dilation, groups=hidden, padding=get_padding(kernel_size, dilation))
 140          )
 141          self.act2 = activation_cls(hidden)
 142          self.pw_project = weight_norm(Conv1d(hidden, channels, 1))  # linear bottleneck, no activation after
 143          for layer in (self.pw_expand, self.dw, self.pw_project):
 144              layer.apply(init_weights)
 145          # Zero-init pw_project so the block starts as an identity no-op
 146          # (residual passthrough), not a random perturbation from step 0 --
 147          # same graft-without-disturbing convention already used in this
 148          # project (MarGan's Stage A/B, flow_block.py's zero-initialized
 149          # `post` layer). weight_norm splits weight into direction
 150          # (weight_v) and magnitude (weight_g) -- zeroing weight_g alone
 151          # makes the effective weight exactly zero regardless of weight_v.
 152          self.pw_project.weight_g.data.zero_()
 153          self.pw_project.bias.data.zero_()
 154  
 155      def forward(self, x: torch.Tensor, x_mask=None) -> torch.Tensor:
 156          xt = self.pw_expand(x)
 157          xt = self.act1(xt)
 158          if x_mask is not None:
 159              xt = xt * x_mask
 160          xt = self.dw(xt)
 161          xt = self.act2(xt)
 162          if x_mask is not None:
 163              xt = xt * x_mask
 164          xt = self.pw_project(xt)
 165          x = xt + x
 166          if x_mask is not None:
 167              x = x * x_mask
 168          return x
 169  
 170      def remove_weight_norm(self):
 171          for layer in (self.pw_expand, self.dw, self.pw_project):
 172              remove_weight_norm(layer)
```

## BanhmiTTS:banhmi_train/vits/utils/normalization.py
SHA-256 `bfd7a6452714229f0d8d1d4469cd9856f62aa2f6f5f535fc3446fadd3f303420`, 82 lines

### lines 25–82
```python
  25  class LeakyReLUActivation(nn.Module):
  26      """Vanilla HiFi-GAN/VITS activation (no Snake at all), matching the
  27      signature of SnakeBeta so it can be used as a drop-in `activation_cls`
  28      -- used by the Generator with the HiFi-GAN resblocks ("1"/"2").
  29      """
  30  
  31      _SLOPE = 0.1
  32  
  33      def __init__(self, channels: int):
  34          super().__init__()
  35  
  36      def forward(self, x: torch.Tensor) -> torch.Tensor:
  37          return F.leaky_relu(x, self._SLOPE)
  38  
  39  
  40  class SnakeBeta(nn.Module):
  41      """Snake-Beta activation (later BigVGAN revision): x + (1/beta) *
  42      sin(alpha * x)^2, with alpha (frequency) and beta (amplitude) as two
  43      *separate* learnable parameters instead of Snake1d's single alpha
  44      reused for both roles.
  45  
  46      Not present in EdgeTTS (which uses plain Snake1d) -- this is a
  47      BanhmiTTS-only addition, used by the Generator with the
  48      inverted-residual resblocks ("mb"/"mrf").
  49      """
  50  
  51      # Bounds alpha/beta to [_min, _max] instead of BigVGAN's usual
  52      # log-space (exp()) reparameterization -- see the TEMPORARY EXPERIMENT
  53      # note below. A hard abs()+clamp() floor has a failure mode log-space
  54      # doesn't: right at the floor, d(1/beta)/dbeta = -1/beta^2, so a floor
  55      # as small as the original 1e-9 turns a single optimizer step that
  56      # pushes beta toward 0 into a ~1e18-magnitude gradient -- enough to
  57      # overflow to Inf under fp16/bf16 autocast well before any global
  58      # grad-norm clip sees it, and Inf/NaN then propagates through every
  59      # downstream layer for the rest of training. 1e-2 keeps 1/beta (and
  60      # its gradient) bounded to a still-generous but finite dynamic range;
  61      # the upper clamp guards the symmetric case where alpha grows
  62      # unbounded and sin(alpha*x)'s gradient (x*cos(alpha*x)) blows up.
  63      _min = 1e-2
  64      _max = 1e2
  65  
  66      def __init__(self, channels: int):
  67          super().__init__()
  68          # TEMPORARY EXPERIMENT (2026-08-26): swapped to EdgeTTS's linear-
  69          # space abs()+clamp() reparameterization (piper_train/vits/modules.py)
  70          # instead of this file's previous log-space exp() one, to test
  71          # whether the reparameterization difference explains a measured
  72          # UTMOS gap. Revert to log_alpha/log_beta + exp() after the test.
  73          # The clamp bounds above were tightened from the original 1e-9
  74          # floor (NaN-prone, see docstring) without reverting the
  75          # experiment itself.
  76          self.alpha = nn.Parameter(torch.ones(1, channels, 1))
  77          self.beta = nn.Parameter(torch.ones(1, channels, 1))
  78  
  79      def forward(self, x: torch.Tensor) -> torch.Tensor:
  80          alpha = self.alpha.abs().clamp(min=self._min, max=self._max)
  81          beta = self.beta.abs().clamp(min=self._min, max=self._max)
  82          return x + (1.0 / beta) * torch.sin(alpha * x) ** 2
```

## BanhmiTTS:banhmi_train/vits/modules/flow_block.py
SHA-256 `e25469c5747d7c106da112c505a752e52468280ef1e940e2336d04aec30e7fd9`, 121 lines

### lines 14–121
```python
  14  class TransformerCouplingLayer(nn.Module):
  15      """VITS2-style residual coupling layer: a WaveNet-style conv stack
  16      predicts an affine transform of one half of the channels from the
  17      other half, with a self-attention pass added for global context.
  18  
  19      Invertibility is unaffected: x1 is still an invertible affine function
  20      of x0 alone; making the *function that computes the affine params*
  21      more expressive doesn't change that x0 passes through unchanged.
  22      """
  23  
  24      def __init__(
  25          self,
  26          channels: int,
  27          hidden_channels: int,
  28          kernel_size: int,
  29          dilation_rate: int,
  30          n_layers: int,
  31          n_heads: int = 2,
  32          p_dropout: float = 0.0,
  33          gin_channels: int = 0,
  34          mean_only: bool = False,
  35      ):
  36          # Local import: attention.py doesn't need to know about flow_block.py,
  37          # only this file needs Encoder, so keep the dependency one-directional.
  38          from ..utils.attention import Encoder
  39  
  40          assert channels % 2 == 0, "channels should be divisible by 2"
  41          super().__init__()
  42          self.half_channels = channels // 2
  43          self.mean_only = mean_only
  44  
  45          self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
  46          self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=p_dropout, gin_channels=gin_channels)
  47          self.attn = Encoder(
  48              hidden_channels, hidden_channels * 2, n_heads=n_heads, n_layers=1, kernel_size=1, p_dropout=p_dropout
  49          )
  50          self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
  51          self.post.weight.data.zero_()
  52          self.post.bias.data.zero_()
  53  
  54      def forward(self, x: torch.Tensor, x_mask: torch.Tensor, g=None, reverse: bool = False):
  55          x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
  56          h = self.pre(x0) * x_mask
  57          # Attention before WaveNet -- matches EdgeTTS's own TransformerCouplingLayer
  58          # and the VITS2 reference implementation's ResidualCouplingTransformersLayer2
  59          # (github.com/p0p4k/vits2_pytorch), verified directly against both.
  60          h = h + self.attn(h, x_mask)
  61          h = self.enc(h, x_mask, g=g)
  62          stats = self.post(h) * x_mask
  63          if self.mean_only:
  64              m, logs = stats, torch.zeros_like(stats)
  65          else:
  66              m, logs = torch.split(stats, [self.half_channels] * 2, 1)
  67  
  68          if not reverse:
  69              x1 = m + x1 * torch.exp(logs) * x_mask
  70              logdet = torch.sum(logs, [1, 2])
  71              return torch.cat([x0, x1], 1), logdet
  72  
  73          x1 = (x1 - m) * torch.exp(-logs) * x_mask
  74          return torch.cat([x0, x1], 1)
  75  
  76      def remove_weight_norm(self):
  77          self.enc.remove_weight_norm()
  78  
  79  
  80  class ResidualCouplingBlock(nn.Module):
  81      def __init__(
  82          self,
  83          channels: int,
  84          hidden_channels: int,
  85          kernel_size: int,
  86          dilation_rate: int,
  87          n_layers: int,
  88          n_flows: int = 4,
  89          gin_channels: int = 0,
  90          n_heads: int = 2,
  91      ):
  92          super().__init__()
  93          self.flows = nn.ModuleList()
  94          for _ in range(n_flows):
  95              self.flows.append(
  96                  TransformerCouplingLayer(
  97                      channels,
  98                      hidden_channels,
  99                      kernel_size,
 100                      dilation_rate,
 101                      n_layers,
 102                      n_heads=n_heads,
 103                      gin_channels=gin_channels,
 104                      mean_only=True,
 105                  )
 106              )
 107              self.flows.append(Flip())
 108  
 109      def forward(self, x: torch.Tensor, x_mask: torch.Tensor, g=None, reverse: bool = False):
 110          flows = self.flows if not reverse else reversed(self.flows)
 111          for flow in flows:
 112              if not reverse:
 113                  x, _ = flow(x, x_mask, g=g, reverse=reverse)
 114              else:
 115                  x = flow(x, x_mask, g=g, reverse=reverse)
 116          return x
 117  
 118      def remove_weight_norm(self):
 119          for flow in self.flows:
 120              if hasattr(flow, "remove_weight_norm"):
 121                  flow.remove_weight_norm()
```

## BanhmiTTS:banhmi_train/vits/modules/synthesizer.py
SHA-256 `c3bc1d98df00d9e895f10a0091d4c6cb75bd9db6e8141cc62b734313f983a404`, 324 lines

### lines 88–140
```python
  88          self.n_speakers = n_speakers
  89          self.use_f0 = use_f0
  90  
  91          self.enc_p = TextEncoder(
  92              n_vocab, inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout
  93          )
  94          if use_vocos:
  95              self.dec = VocosGenerator(
  96                  inter_channels,
  97                  n_fft=vocos_n_fft,
  98                  hop_length=vocos_hop_length,
  99                  dim=vocos_dim,
 100                  intermediate_dim=vocos_intermediate_dim,
 101                  num_layers=vocos_num_layers,
 102                  gin_channels=gin_channels,
 103                  use_f0=use_f0,
 104              )
 105          else:
 106              self.dec = Generator(
 107                  inter_channels,
 108                  resblock,
 109                  resblock_kernel_sizes,
 110                  resblock_dilation_sizes,
 111                  upsample_rates,
 112                  upsample_initial_channel,
 113                  upsample_kernel_sizes,
 114                  gin_channels=gin_channels,
 115                  mb_expansion=mb_expansion,
 116              )
 117          self.enc_q = PosteriorEncoder(
 118              spec_channels,
 119              inter_channels,
 120              hidden_channels,
 121              posterior_encoder_kernel_size,
 122              posterior_encoder_dilation_rate,
 123              posterior_encoder_layers,
 124              gin_channels=gin_channels,
 125          )
 126          self.flow = ResidualCouplingBlock(
 127              inter_channels,
 128              hidden_channels,
 129              flow_kernel_size,
 130              flow_dilation_rate,
 131              n_layers=4,  # matches EdgeTTS's hardcoded WN depth per coupling layer
 132              n_flows=flow_n_flows,
 133              gin_channels=gin_channels,
 134          )
 135  
 136          self.dp = StochasticDurationPredictor(hidden_channels, 3, 0.5, 4, gin_channels=gin_channels)
 137  
 138          if use_f0:
 139              self.f0_predictor = F0Predictor(hidden_channels, 256, 3, 0.5)
 140  
```

### lines 216–241
```python
 216      def _monotonic_alignment(self, x_mask, y_mask, z_p, m_p, logs_p, mas_noise_scale: float = 0.0):
 217          """Finds the most likely phoneme<->frame alignment given the
 218          current prior, via the negative cross-entropy between each frame's
 219          posterior sample and each phoneme's prior distribution.
 220  
 221          mas_noise_scale (VITS2's "noise-scaled MAS"): early in training the
 222          prior/posterior are poorly calibrated, so the argmax alignment can
 223          lock onto a bad path and never recover (monotonic search has no way
 224          to "undo" a bad early commitment). Injecting noise proportional to
 225          neg_cent's own spread lets a few different alignments win early on;
 226          the caller anneals this to 0 over training so it converges to the
 227          exact deterministic search vanilla VITS always used.
 228          """
 229          s_p_sq_r = torch.exp(-2 * logs_p)  # [b, d, t_s]
 230          neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True)  # [b, 1, t_s]
 231          neg_cent2 = torch.matmul(-0.5 * (z_p**2).transpose(1, 2), s_p_sq_r)  # [b, t_t, t_s]
 232          neg_cent3 = torch.matmul(z_p.transpose(1, 2), m_p * s_p_sq_r)  # [b, t_t, t_s]
 233          neg_cent4 = torch.sum(-0.5 * (m_p**2) * s_p_sq_r, [1], keepdim=True)  # [b, 1, t_s]
 234          neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4
 235  
 236          if mas_noise_scale > 0:
 237              neg_cent = neg_cent + torch.randn_like(neg_cent) * torch.std(neg_cent) * mas_noise_scale
 238  
 239          attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
 240          return maximum_path(neg_cent, attn_mask.squeeze(1)).unsqueeze(1).detach()
 241  
```

### lines 250–324
```python
 250          max_len=None,
 251          onnx_export: bool = False,
 252      ):
 253          x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
 254          g = self._speaker_embedding(sid)
 255  
 256          logw = self.dp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w)
 257          # Same guard/bound as flows.py's Log._EXP_CLAMP_MAX: clamp before
 258          # exp() rather than only handling the fallout after. This exp() was
 259          # previously unguarded -- observed root-causing a full model_g/
 260          # model_d NaN collapse (StochasticDurationPredictor's reverse-mode
 261          # sampling, called every validation epoch via _log_audio_samples):
 262          # an unclamped logw let exp(logw) reach a very large but still-finite
 263          # value that survived this line, then blew up further downstream
 264          # (mask multiply, w_ceil sum, sequence_mask) into inf/NaN. Clamping
 265          # logw itself keeps w_ceil bounded to a generous-but-sane frame
 266          # count (exp(10) ~= 22000 frames) instead of merely reacting to
 267          # non-finite values after the fact (see the nan_to_num/degenerate
 268          # handling below, which stays as a second line of defense).
 269          w_ceil = torch.ceil(torch.exp(logw.clamp(max=10.0)) * x_mask * length_scale)
 270          # An undertrained/unstable duration predictor can occasionally emit a
 271          # non-finite logw (exp() overflow -> inf, or NaN) -> casting inf/NaN
 272          # to long is undefined (observed: wraps to a large negative number),
 273          # which then crashes commons.sequence_mask's torch.arange on a
 274          # negative length. nan_to_num first (clamp alone lets NaN through
 275          # unchanged, since comparisons against NaN are always false), then
 276          # clamp to a generous-but-finite frame count to keep infer() crash-safe.
 277          y_lengths = torch.sum(w_ceil, [1, 2])
 278          # Surface duration-predictor collapse the moment it happens (rather
 279          # than silently falling back to a near-empty 1-frame output with no
 280          # trace) -- non-finite or near-zero durations here mean self.dp's
 281          # reverse-mode sampling has degenerated, independently of how good
 282          # loss_mel/val_loss_mel look (those never exercise this code path).
 283          degenerate = ~torch.isfinite(y_lengths) | (y_lengths < 4)
 284          if degenerate.any():
 285              _LOGGER.warning(
 286                  "StochasticDurationPredictor produced degenerate duration(s) in infer(): "
 287                  "raw summed w_ceil=%s (finite min=%s over the non-degenerate rest) -- "
 288                  "falling back to a 1-frame minimum for the affected batch item(s). "
 289                  "This indicates dp's reverse-sampling has collapsed, not a Generator/mel issue.",
 290                  y_lengths.detach().cpu().tolist(),
 291                  y_lengths[~degenerate].min().item() if (~degenerate).any() else "n/a",
 292              )
 293          y_lengths = torch.nan_to_num(y_lengths, nan=1.0, posinf=100000.0, neginf=1.0)
 294          y_lengths = torch.clamp(y_lengths, min=1, max=100000).long()
 295          y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y_lengths.max()), 1).type_as(x_mask)
 296          attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
 297          attn = commons.generate_path(w_ceil, attn_mask)
 298  
 299          attn_sq = attn.squeeze(1)
 300          # [b, t', t] x [b, t, d] -> [b, d, t']: expand per-phoneme prior to per-frame
 301          m_p = torch.matmul(attn_sq, m_p.transpose(1, 2)).transpose(1, 2)
 302          logs_p = torch.matmul(attn_sq, logs_p.transpose(1, 2)).transpose(1, 2)
 303  
 304          f0_frame = None
 305          if self.use_f0:
 306              log_f0_pred = self.f0_predictor(x, x_mask)  # [b, 1, t_s]
 307              # Same hard-alignment expansion as forward()'s f0_frame, and the
 308              # same clamp-before-exp guard as logw above (this exp() has the
 309              # identical overflow failure mode).
 310              f0_frame = torch.matmul(attn_sq, log_f0_pred.transpose(1, 2)).transpose(1, 2)  # [b, 1, t_t]
 311              f0_frame = torch.exp(f0_frame.clamp(max=10.0))
 312  
 313          z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
 314          z = self.flow(z_p, y_mask, g=g, reverse=True)
 315          if isinstance(self.dec, VocosGenerator):
 316              # onnx_export routes ISTFTHead around torch.complex/torch.fft
 317              # (unsupported by torch.onnx's exporter) -- a no-op kwarg for
 318              # every other Generator family, which never uses either op.
 319              f0_out = f0_frame[:, :, :max_len] if f0_frame is not None else None
 320              o = self.dec((z * y_mask)[:, :, :max_len], g=g, f0=f0_out, onnx_export=onnx_export)
 321          else:
 322              o = self.dec((z * y_mask)[:, :, :max_len], g=g)
 323  
 324          return o, attn, y_mask, (z, z_p, m_p, logs_p)
```

## BanhmiTTS:banhmi_train/vits/training.py
SHA-256 `c61baedeb0f39748d5beee50d2a0df558189c044a69bc233a9c512589962c13c`, 790 lines

### lines 195–220
```python
 195              vocos_intermediate_dim=self.hparams.vocos_intermediate_dim,
 196              vocos_num_layers=self.hparams.vocos_num_layers,
 197              use_f0=self.hparams.use_f0,
 198          )
 199          if self.hparams.use_vocos:
 200              self.model_d = VocosMultiPeriodDiscriminator()
 201              self.model_d_mrd = VocosMultiResolutionDiscriminator()
 202          else:
 203              self.model_d = MultiPeriodDiscriminator(use_spectral_norm=self.hparams.use_spectral_norm)
 204              self.model_d_mrd = (
 205                  MultiResolutionDiscriminator(use_spectral_norm=self.hparams.use_spectral_norm)
 206                  if self.hparams.resblock in INVERTED_RESBLOCKS
 207                  else None
 208              )
 209          self.model_d_dur = DurationDiscriminator(
 210              in_channels=self.hparams.hidden_channels,
 211              filter_channels=self.hparams.hidden_channels,
 212              kernel_size=3,
 213              p_dropout=self.hparams.p_dropout,
 214              gin_channels=_GIN_CHANNELS,
 215          )
 216  
 217          self._train_dataset: Optional[Dataset] = None
 218          self._val_dataset: Optional[Dataset] = None
 219          self._test_dataset: Optional[Dataset] = None
 220          self._audio_sample_dataset: Optional[Dataset] = None
```

### lines 436–480
```python
 436      def optimizer_step(
 437          self, epoch, batch_idx, optimizer, optimizer_idx=0, optimizer_closure=None, **kwargs
 438      ):
 439          """Discards an update whose gradients aren't finite instead of letting
 440          it reach the optimizer.
 441  
 442          This is the safety net `torch.cuda.amp.GradScaler` provides for free
 443          under fp16 (check for inf/NaN, skip the step, carry on) and which
 444          bf16 training silently does without, since bf16 needs no loss scaling
 445          and therefore gets no scaler. Without it, one non-finite gradient is
 446          permanent: AdamW writes NaN into the parameters, and every subsequent
 447          forward pass through those parameters is NaN forever after. That is
 448          exactly how StochasticDurationPredictor kept dying here -- and because
 449          dp's gradients don't reach dec/enc_p, loss_mel went on improving and
 450          hid the damage for hundreds of epochs.
 451  
 452          Skipping costs one finite-check per parameter per step and throws away
 453          only the genuinely broken batch; the parameters keep their last good
 454          values and training continues from there.
 455  
 456          Under DDP this stays consistent across ranks without extra
 457          synchronisation: gradients are all-reduced during backward, and a NaN
 458          on any rank propagates through that averaging, so every rank sees the
 459          same non-finite gradients and independently reaches the same decision.
 460          """
 461          optimizer_closure()  # forward + backward; populates .grad
 462  
 463          nonfinite = False
 464          for group in optimizer.param_groups:
 465              for param in group["params"]:
 466                  if param.grad is not None and not torch.isfinite(param.grad).all():
 467                      nonfinite = True
 468                      break
 469              if nonfinite:
 470                  break
 471  
 472          if optimizer_idx == 0:
 473              self._total_steps += 1
 474  
 475          if not nonfinite:
 476              optimizer.step()
 477              return
 478  
 479          optimizer.zero_grad(set_to_none=True)
 480          self._skipped_steps += 1
```

### lines 520–620
```python
 520              _attn,
 521              ids_slice,
 522              x_mask,
 523              z_mask,
 524              (_z, z_p, m_p, logs_p, _m_q, logs_q),
 525              (x_hidden, logw, logw_, l_f0),
 526          ) = self.model_g(x, x_lengths, spec, spec_lengths, mas_noise_scale=mas_noise_scale, f0=batch.f0s)
 527          self._y_hat = y_hat
 528          self.log("mas_noise_scale", mas_noise_scale)
 529  
 530          # Saved for training_step_d's duration discriminator
 531          self._dur_x, self._dur_mask, self._dur_real, self._dur_fake = x_hidden, x_mask, logw_, logw
 532  
 533          mel = spec_to_mel_torch(
 534              spec, self.hparams.filter_length, self.hparams.mel_channels,
 535              self.hparams.sample_rate, self.hparams.mel_fmin, self.hparams.mel_fmax,
 536          )
 537          y_mel = slice_segments(mel, ids_slice, self.hparams.segment_size // self.hparams.hop_length)
 538          y_hat_mel = mel_spectrogram_torch(
 539              y_hat.squeeze(1), self.hparams.filter_length, self.hparams.mel_channels,
 540              self.hparams.sample_rate, self.hparams.hop_length, self.hparams.win_length,
 541              self.hparams.mel_fmin, self.hparams.mel_fmax,
 542          )
 543          y = slice_segments(y, ids_slice * self.hparams.hop_length, self.hparams.segment_size)
 544          self._y = y
 545  
 546          _y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.model_d(y, y_hat)
 547          _disc_loss_fn, gen_loss_fn, feat_loss_fn = self._gan_loss_fns()
 548  
 549          with autocast(self.device.type, enabled=False):
 550              loss_dur = torch.sum(l_length.float())
 551              loss_mel = F.l1_loss(y_mel, y_hat_mel) * self.hparams.c_mel
 552              loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * self.hparams.c_kl
 553  
 554              loss_fm = feat_loss_fn(fmap_r, fmap_g)
 555              loss_gen, _ = gen_loss_fn(y_d_hat_g)
 556              loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl + l_f0
 557              self.log("loss_f0", l_f0)
 558  
 559              # Stashed so validation_step can log a *validation-only*
 560              # "val_loss_mel" (see below) -- self.log("loss_mel", ...) alone
 561              # mixes training-step and validation-step calls into one epoch
 562              # average, which is a noisier/less meaningful "best model" signal
 563              # than a clean non-adversarial validation reconstruction loss
 564              # (matches EdgeTTS's own val_loss_mel).
 565              self._last_loss_mel = loss_mel.detach()
 566              self.log("loss_mel", loss_mel)
 567              self.log("loss_kl", loss_kl)
 568              self.log("loss_dur", loss_dur)
 569              self.log("loss_gen", loss_gen)
 570              self.log("loss_fm", loss_fm)
 571  
 572              if self.model_d_mrd is not None:
 573                  _y_d_hat_r_mrd, y_d_hat_g_mrd, fmap_r_mrd, fmap_g_mrd = self.model_d_mrd(y, y_hat)
 574                  loss_fm_mrd = feat_loss_fn(fmap_r_mrd, fmap_g_mrd)
 575                  loss_gen_mrd, _ = gen_loss_fn(y_d_hat_g_mrd)
 576                  # Only Vocos's own recipe down-weights MRD's contribution
 577                  # (vocos/experiment.py: loss_gen_mp + mrd_loss_coeff *
 578                  # loss_gen_mrd) -- the HiFi-GAN-family MRD keeps its
 579                  # original equal-weight sum (coeff=1.0).
 580                  mrd_coeff = self.hparams.vocos_mrd_loss_coeff if self.hparams.use_vocos else 1.0
 581                  loss_gen_all = loss_gen_all + mrd_coeff * (loss_gen_mrd + loss_fm_mrd)
 582                  self.log("loss_gen_mrd", loss_gen_mrd)
 583  
 584              _dur_probs_r, dur_probs_hat = self.model_d_dur(x_hidden, x_mask, logw_, logw)
 585              loss_dur_gen, _ = generator_loss(dur_probs_hat)
 586              loss_gen_all = loss_gen_all + loss_dur_gen
 587              self.log("loss_dur_gen", loss_dur_gen)
 588  
 589              self.log("loss_gen_all", loss_gen_all)
 590              return loss_gen_all
 591  
 592      def training_step_d(self, batch: Batch):
 593          y, y_hat = self._y, self._y_hat
 594          y_d_hat_r, y_d_hat_g, _, _ = self.model_d(y, y_hat.detach())
 595          disc_loss_fn, _gen_loss_fn, _feat_loss_fn = self._gan_loss_fns()
 596  
 597          with autocast(self.device.type, enabled=False):
 598              loss_disc, *_ = disc_loss_fn(y_d_hat_r, y_d_hat_g)
 599              loss_disc_all = loss_disc
 600              self.log("loss_disc", loss_disc)
 601  
 602              if self.model_d_mrd is not None:
 603                  y_d_hat_r_mrd, y_d_hat_g_mrd, _, _ = self.model_d_mrd(y, y_hat.detach())
 604                  loss_disc_mrd, *_ = disc_loss_fn(y_d_hat_r_mrd, y_d_hat_g_mrd)
 605                  mrd_coeff = self.hparams.vocos_mrd_loss_coeff if self.hparams.use_vocos else 1.0
 606                  loss_disc_all = loss_disc_all + mrd_coeff * loss_disc_mrd
 607                  self.log("loss_disc_mrd", loss_disc_mrd)
 608  
 609              dur_probs_r, dur_probs_hat = self.model_d_dur(
 610                  self._dur_x.detach(), self._dur_mask, self._dur_real.detach(), self._dur_fake.detach()
 611              )
 612              loss_disc_dur, *_ = discriminator_loss(dur_probs_r, dur_probs_hat)
 613              loss_disc_all = loss_disc_all + loss_disc_dur
 614              self.log("loss_disc_dur", loss_disc_dur)
 615  
 616              self.log("loss_disc_all", loss_disc_all)
 617              return loss_disc_all
 618  
 619      def validation_step(self, batch: Batch, batch_idx: int):
 620          val_loss = self.training_step_g(batch) + self.training_step_d(batch)
```

### lines 698–730
```python
 698      def _make_adamw(params, **kwargs) -> torch.optim.AdamW:
 699          # fused=True dispatches to a single fused CUDA kernel for the whole
 700          # parameter-update step instead of one kernel launch per tensor --
 701          # meaningful here with 72M+ params split across two optimizers.
 702          # Only on torch>=2.0 (the `fused` kwarg doesn't exist before that);
 703          # falls back silently on anything else (older torch, CPU-only, etc).
 704          try:
 705              return torch.optim.AdamW(params, fused=torch.cuda.is_available(), **kwargs)
 706          except TypeError:
 707              return torch.optim.AdamW(params, **kwargs)
 708  
 709      def configure_optimizers(self):
 710          discriminators = [self.model_d, self.model_d_mrd, self.model_d_dur]
 711          discriminator_params = itertools.chain(
 712              *(d.parameters() for d in discriminators if d is not None)
 713          )
 714          optimizers = [
 715              self._make_adamw(
 716                  self.model_g.parameters(), lr=self.hparams.learning_rate,
 717                  betas=self.hparams.betas, eps=self.hparams.eps,
 718              ),
 719              self._make_adamw(
 720                  discriminator_params, lr=self.hparams.learning_rate,
 721                  betas=self.hparams.betas, eps=self.hparams.eps,
 722              ),
 723          ]
 724          schedulers = [
 725              torch.optim.lr_scheduler.ExponentialLR(optimizers[0], gamma=self.hparams.lr_decay),
 726              torch.optim.lr_scheduler.ExponentialLR(optimizers[1], gamma=self.hparams.lr_decay),
 727          ]
 728          return optimizers, schedulers
 729  
 730      @staticmethod
```

## BanhmiTTS:banhmi_train/vits/utils/flows.py
SHA-256 `b1a9288187dc1401e3178781f6b67721d61719330890d873a0b29c401aadfa05`, 134 lines

### lines 1–134
```python
   1  """Invertible flow steps used to build up the normalizing-flow prior
   2  (ResidualCouplingBlock in flow_block.py) and the stochastic duration
   3  predictor's two flow stacks.
   4  
   5  Each flow's forward() returns (output, logdet) when reverse=False and just
   6  output when reverse=True -- the log-determinant of the Jacobian is only
   7  needed for the forward (density-evaluation) direction.
   8  """
   9  import math
  10  
  11  import torch
  12  from torch import nn
  13  
  14  from .dds_conv import DDSConv
  15  from .transforms import piecewise_rational_quadratic_transform
  16  
  17  
  18  class Log(nn.Module):
  19      # Shared with ElementwiseAffine's bound below -- exp(10) ~= 2.2e4, still
  20      # far more than any legitimate value this flow produces (log-duration
  21      # of 10 would mean a duration of ~22000 frames for one phoneme), but
  22      # tight enough that a value passing this guard can't itself explode a
  23      # downstream Conv1d stack (the previous bound of 30 => exp(30) ~= 1e13
  24      # was observed to survive this guard intact and then blow up further
  25      # downstream, causing a 3rd StochasticDurationPredictor collapse despite
  26      # both this guard and gradient clipping being active).
  27      _EXP_CLAMP_MAX = 10.0
  28  
  29      def forward(self, x: torch.Tensor, x_mask: torch.Tensor, reverse: bool = False):
  30          if not reverse:
  31              y = torch.log(torch.clamp_min(x, 1e-5)) * x_mask
  32              logdet = torch.sum(-y, [1, 2])
  33              return y, logdet
  34          # reverse=True runs every training step too (SynthesizerTrn.forward's
  35          # duration-discriminator sample, StochasticDurationPredictor._sample),
  36          # not just at inference -- an ordinary-looking upstream value here can
  37          # still exp() into inf under bf16/fp32 if training pushes x large
  38          # before this flow stabilizes. Same failure family as the spline
  39          # transform's unguarded division/log (see transforms.py's _EPS) --
  40          # ported unchanged from upstream Piper/EdgeTTS, which has this same
  41          # gap (piper_train/vits/modules.py's Log/ElementwiseAffine).
  42          return torch.exp(x.clamp(max=self._EXP_CLAMP_MAX)) * x_mask
  43  
  44  
  45  class Flip(nn.Module):
  46      def forward(self, x: torch.Tensor, *args, reverse: bool = False, **kwargs):
  47          x = torch.flip(x, [1])
  48          if not reverse:
  49              return x, torch.zeros(x.size(0)).type_as(x)
  50          return x
  51  
  52  
  53  class ElementwiseAffine(nn.Module):
  54      # self.logs is an unconstrained learnable parameter -- nothing bounds how
  55      # far gradient descent can push it. Both branches below exponentiate it
  56      # (or its negation), so an unclamped logs that drifts too far either way
  57      # overflows exp() to inf under bf16/fp32, which then turns into NaN the
  58      # moment it multiplies a zero (e.g. via x_mask's padding). Observed in
  59      # practice: a baseline (non-Vocos) run's dp.flows/post_flows collapsed to
  60      # NaN across every model_g/model_d parameter within a few epochs after a
  61      # resume, root-caused to here -- not present in upstream Piper/EdgeTTS
  62      # either (piper_train/vits/modules.py has the identical unguarded code),
  63      # so this is a latent bug inherited from there, not a regression.
  64      # +-10 keeps exp(logs)/exp(-logs) within [~4.5e-5, ~2.2e4] -- wide enough
  65      # to never bind a legitimately-learned scale, per Log's clamp above.
  66      _LOGS_CLAMP = 10.0
  67  
  68      def __init__(self, channels: int):
  69          super().__init__()
  70          self.m = nn.Parameter(torch.zeros(channels, 1))
  71          self.logs = nn.Parameter(torch.zeros(channels, 1))
  72  
  73      def forward(self, x: torch.Tensor, x_mask: torch.Tensor, reverse: bool = False, **kwargs):
  74          logs = self.logs.clamp(min=-self._LOGS_CLAMP, max=self._LOGS_CLAMP)
  75          if not reverse:
  76              y = (self.m + torch.exp(logs) * x) * x_mask
  77              logdet = torch.sum(logs * x_mask, [1, 2])
  78              return y, logdet
  79          return (x - self.m) * torch.exp(-logs) * x_mask
  80  
  81  
  82  class ConvFlow(nn.Module):
  83      """Coupling layer whose affine-transform-predictor output instead
  84      parameterizes a rational-quadratic spline (a strictly more expressive,
  85      still-invertible transform of x1)."""
  86  
  87      def __init__(
  88          self,
  89          in_channels: int,
  90          filter_channels: int,
  91          kernel_size: int,
  92          n_layers: int,
  93          num_bins: int = 10,
  94          tail_bound: float = 5.0,
  95      ):
  96          super().__init__()
  97          self.half_channels = in_channels // 2
  98          self.filter_channels = filter_channels
  99          self.num_bins = num_bins
 100          self.tail_bound = tail_bound
 101  
 102          self.pre = nn.Conv1d(self.half_channels, filter_channels, 1)
 103          self.convs = DDSConv(filter_channels, kernel_size, n_layers, p_dropout=0.0)
 104          self.proj = nn.Conv1d(filter_channels, self.half_channels * (num_bins * 3 - 1), 1)
 105          self.proj.weight.data.zero_()
 106          self.proj.bias.data.zero_()
 107  
 108      def forward(self, x: torch.Tensor, x_mask: torch.Tensor, g=None, reverse: bool = False):
 109          x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
 110          h = self.convs(self.pre(x0), x_mask, g=g)
 111          h = self.proj(h) * x_mask
 112  
 113          b, c, t = x0.shape
 114          h = h.reshape(b, c, -1, t).permute(0, 1, 3, 2)  # [b, c*?, t] -> [b, c, t, ?]
 115  
 116          unnormalized_widths = h[..., : self.num_bins] / math.sqrt(self.filter_channels)
 117          unnormalized_heights = h[..., self.num_bins : 2 * self.num_bins] / math.sqrt(self.filter_channels)
 118          unnormalized_derivatives = h[..., 2 * self.num_bins :]
 119  
 120          x1, logabsdet = piecewise_rational_quadratic_transform(
 121              x1,
 122              unnormalized_widths,
 123              unnormalized_heights,
 124              unnormalized_derivatives,
 125              inverse=reverse,
 126              tails="linear",
 127              tail_bound=self.tail_bound,
 128          )
 129  
 130          x = torch.cat([x0, x1], 1) * x_mask
 131          logdet = torch.sum(logabsdet * x_mask, [1, 2])
 132          if not reverse:
 133              return x, logdet
 134          return x
```

## BanhmiTTS:configs/banhmi_mrf.yaml
SHA-256 `85d468c4e22313d8cab0a9648fb61df70e513a81cc23650c0a8aef2db194c2ed`, 69 lines

### lines 1–69
```yaml
   1  # MRF variant of the mbconv novelty direction (configs/banhmi_mbconv.yaml):
   2  # HiFi-GAN's *real* MRF mechanism (N branches run in PARALLEL on the same
   3  # input, summed then divided by N -- what resblock "1"/"2" already do) but
   4  # each branch is a ResBlockInverted (MobileNetV2-style inverted-residual,
   5  # see banhmi_train/vits/utils/resblocks.py) instead of ResBlock1/2's dilated
   6  # convs. resblock="mrf" in banhmi_train/vits/modules/generator.py.
   7  #
   8  # Picked over plain sequential stacking (resblock="mb",
   9  # configs/banhmi_mbconv.yaml) and over the other 3 MRF expansion schedules
  10  # tried, per the full sweep in docs/mbconv_resblock_report.md SS6-SS9:
  11  # expansion=(1,1,1) is the only MRF schedule that stays faster than plain
  12  # HiFi-GAN resblock "2" in isolated generator benchmarks (paired-ratio
  13  # methodology, ~0.43x baseline's time) -- (2,2,1)/(4,4,1)/(6,6,1) all lose
  14  # that speed edge once 3 parallel branches replace 1-2 sequential blocks.
  15  # Quality-wise MRF(1,1,1) (loss_mel=9.0788, 800-step overfit) is close to
  16  # but not better than sequential resblock="mb" at the same expansion
  17  # (8.9742) -- the case for MRF over sequential here is mainly HiFi-GAN
  18  # design-fidelity (real multi-receptive-field mechanism) plus the speed
  19  # margin, not a clean quality win. See the report for the full comparison
  20  # and the noise caveats on the speed numbers.
  21  #
  22  # Usage:
  23  #   python -m banhmi_train.train --dataset-dir <dir> --config configs/banhmi_mrf.yaml --batch-size <N> --max-epochs <N> ...
  24  
  25  # TextEncoder / PosteriorEncoder / Flow / DurationPredictor width -- same as
  26  # configs/default.yaml, unchanged by this novelty (generator-only change).
  27  hidden_channels: 192
  28  inter_channels: 192
  29  filter_channels: 768
  30  n_heads: 2
  31  n_layers: 6
  32  kernel_size: 3
  33  p_dropout: 0.1
  34  
  35  # Generator: resblock "mrf" selects the parallel-branch ResBlockInverted
  36  # path (see Generator's own comment for the mechanism). mb_mrf_kernel_sizes
  37  # defaults to (3,5,7) inside Generator -- matches resblock_kernel_sizes
  38  # below, not repeated here as a separate knob. resblock_kernel_sizes/
  39  # resblock_dilation_sizes below are unused in "mrf" mode (Generator still
  40  # requires them positionally) -- left at the medium-tier values for when
  41  # switching back to resblock "2".
  42  resblock: "mrf"
  43  mb_expansion: [1, 1, 1]
  44  resblock_kernel_sizes: [3, 5, 7]
  45  resblock_dilation_sizes: [[1, 2], [2, 6], [3, 12]]
  46  upsample_rates: [8, 8, 4]
  47  upsample_initial_channel: 256
  48  upsample_kernel_sizes: [16, 16, 8]
  49  
  50  # SnakeBeta activations (throughout the Generator) and the MRD discriminator
  51  # come with the inverted-residual resblock automatically -- see
  52  # INVERTED_RESBLOCKS in banhmi_train/vits/modules/generator.py.
  53  
  54  # Flow (ResidualCouplingBlock) -- always VITS2-style transformer-in-flow,
  55  # no plain-WN-flow toggle exists anymore.
  56  flow_kernel_size: 5
  57  flow_dilation_rate: 1
  58  flow_n_flows: 4
  59  
  60  # VITS2's noise-scaled MAS -- always active (no toggle), anneals from
  61  # initial to 0 over training via these two knobs.
  62  mas_noise_scale_initial: 0.01
  63  mas_noise_scale_decay: 0.000002
  64  
  65  # PosteriorEncoder (training-only, discarded before ONNX export) -- matches
  66  # EdgeTTS/vanilla Piper's own hardcoded WaveNet depth of 16.
  67  posterior_encoder_kernel_size: 5
  68  posterior_encoder_dilation_rate: 1
  69  posterior_encoder_layers: 16
```

## BanhmiTTS:configs/banhmi_mb_sequential_111.yaml
SHA-256 `6840425fc043bc619a2f92dc37eeafd6f9a23bfbae10a53a8446b749f0fcaa62`, 65 lines

### lines 1–65
```yaml
   1  # Sequential (resblock="mb") variant of ResBlockInverted, expansion=(1,1,1)
   2  # -- the schedule that won the overfit-test sweep for the SEQUENTIAL
   3  # (2 blocks stacked, not 3 parallel branches) stacking style, revisited
   4  # after finding MRF(1,1,1)'s parallel-branch RTF disadvantage on real
   5  # hardware (ONNX Runtime/VNNI, long audio) -- see docs/TODO.md item 1 and
   6  # docs/final_4way_comparison_report.md SS3.3. Quick forward-pass-only
   7  # benchmark (untrained) showed sequential staying faster than baseline at
   8  # every tested sequence length (T=32..700), unlike MRF-parallel which
   9  # crosses over to slower at long T -- this config trains it to full 1500
  10  # epochs to confirm with a real, converged checkpoint.
  11  #
  12  # Uses canonical_split.json (training_output/ljspeech/medium/) so this run
  13  # gets the EXACT SAME train/val/test partition as baseline_v2_noclip --
  14  # fixes the train/test-leakage-via-random_split bug at the root (see
  15  # docs/mrf_full_investigation_report.md SS8.2) instead of reconstructing a
  16  # separate correct-for-this-model-only test set after the fact.
  17  #
  18  # Usage:
  19  #   python -m banhmi_train.train --dataset-dir <dir> --config configs/banhmi_mb_sequential_111.yaml --batch-size <N> --max-epochs <N> ...
  20  
  21  # TextEncoder / PosteriorEncoder / Flow / DurationPredictor width -- same as
  22  # configs/default.yaml, unchanged by this novelty (generator-only change).
  23  hidden_channels: 192
  24  inter_channels: 192
  25  filter_channels: 768
  26  n_heads: 2
  27  n_layers: 6
  28  kernel_size: 3
  29  p_dropout: 0.1
  30  
  31  # Generator: resblock "mb" selects sequential-stacking ResBlockInverted
  32  # (see Generator's own comment / mbconv_resblock_report.md for the block
  33  # itself). mb_kernel_size/mb_blocks_per_stage left at Generator's own
  34  # validated defaults (kernel=3, blocks_per_stage=2) -- not repeated here as
  35  # separate knobs. resblock_kernel_sizes/resblock_dilation_sizes below are
  36  # unused in "mb" mode (Generator still requires them positionally) -- left
  37  # at the medium-tier values for when switching back to resblock "2".
  38  resblock: "mb"
  39  mb_expansion: [1, 1, 1]
  40  resblock_kernel_sizes: [3, 5, 7]
  41  resblock_dilation_sizes: [[1, 2], [2, 6], [3, 12]]
  42  upsample_rates: [8, 8, 4]
  43  upsample_initial_channel: 256
  44  upsample_kernel_sizes: [16, 16, 8]
  45  
  46  # SnakeBeta activations (throughout the Generator) and the MRD discriminator
  47  # come with the inverted-residual resblock automatically -- see
  48  # INVERTED_RESBLOCKS in banhmi_train/vits/modules/generator.py.
  49  
  50  # Flow (ResidualCouplingBlock) -- always VITS2-style transformer-in-flow,
  51  # no plain-WN-flow toggle exists anymore.
  52  flow_kernel_size: 5
  53  flow_dilation_rate: 1
  54  flow_n_flows: 4
  55  
  56  # VITS2's noise-scaled MAS -- always active (no toggle), anneals from
  57  # initial to 0 over training via these two knobs.
  58  mas_noise_scale_initial: 0.01
  59  mas_noise_scale_decay: 0.000002
  60  
  61  # PosteriorEncoder (training-only, discarded before ONNX export) -- matches
  62  # EdgeTTS/vanilla Piper's own hardcoded WaveNet depth of 16.
  63  posterior_encoder_kernel_size: 5
  64  posterior_encoder_dilation_rate: 1
  65  posterior_encoder_layers: 16
```

## BanhmiTTS:configs/default.yaml
SHA-256 `7e724b00aefb9db68ef7a7cd56eb98f034553b022a5958b3e78cc529798dc596`, 52 lines

### lines 1–52
```yaml
   1  # Pinned snapshot of banhmi_train's own code defaults (VitsModel.__init__ in
   2  # vits/training.py) -- i.e. what you get running `python -m banhmi_train.train
   3  # --quality medium` with NO --config at all. Every value below is spelled out
   4  # explicitly (not left to the code's own defaults), so this file stays a
   5  # trustworthy reference even if those defaults ever drift, and gives you a
   6  # named, diffable config to pass around instead of "no --config" (which is
   7  # easy to get wrong/forget).
   8  #
   9  # This is BanhmiTTS's actual baseline identity: the full VITS2 recipe
  10  # (transformer-in-flow coupling + duration discriminator + noise-scaled MAS)
  11  # is baked permanently into the architecture -- not a toggle, always on --
  12  # since the project builds on VITS2 rather than replicating EdgeTTS's
  13  # opt-in --use-vits2. With the HiFi-GAN resblock "2" below, the Generator
  14  # uses LeakyReLU and training uses MPD only (no MRD) -- that pairing is
  15  # fixed by the resblock type, not a separate setting.
  16  #
  17  # Usage:
  18  #   python -m banhmi_train.train --dataset-dir <dir> --config configs/default.yaml --batch-size <N> ...
  19  
  20  # TextEncoder / PosteriorEncoder / Flow / DurationPredictor width
  21  hidden_channels: 192
  22  inter_channels: 192
  23  filter_channels: 768
  24  n_heads: 2
  25  n_layers: 6
  26  kernel_size: 3
  27  p_dropout: 0.1
  28  
  29  # Generator (`--quality medium` tier -- lighter resblock, 3 upsample stages)
  30  resblock: "2"
  31  resblock_kernel_sizes: [3, 5, 7]
  32  resblock_dilation_sizes: [[1, 2], [2, 6], [3, 12]]
  33  upsample_rates: [8, 8, 4]
  34  upsample_initial_channel: 256
  35  upsample_kernel_sizes: [16, 16, 8]
  36  
  37  # Flow (ResidualCouplingBlock) -- always VITS2-style transformer-in-flow,
  38  # no plain-WN-flow toggle exists anymore.
  39  flow_kernel_size: 5
  40  flow_dilation_rate: 1
  41  flow_n_flows: 4
  42  
  43  # VITS2's noise-scaled MAS -- always active (no toggle), anneals from
  44  # initial to 0 over training via these two knobs.
  45  mas_noise_scale_initial: 0.01
  46  mas_noise_scale_decay: 0.000002
  47  
  48  # PosteriorEncoder (training-only, discarded before ONNX export) -- matches
  49  # EdgeTTS/vanilla Piper's own hardcoded WaveNet depth of 16.
  50  posterior_encoder_kernel_size: 5
  51  posterior_encoder_dilation_rate: 1
  52  posterior_encoder_layers: 16
```

## piper:models.py
SHA-256 `55f024fecff9068ec1f9136ac428e29dd56c29f30d01e9d1eb2aaf8e54aeb71f`, 732 lines

### lines 212–256
```python
 212  class ResidualCouplingBlock(nn.Module):
 213      def __init__(
 214          self,
 215          channels: int,
 216          hidden_channels: int,
 217          kernel_size: int,
 218          dilation_rate: int,
 219          n_layers: int,
 220          n_flows: int = 4,
 221          gin_channels: int = 0,
 222      ):
 223          super().__init__()
 224          self.channels = channels
 225          self.hidden_channels = hidden_channels
 226          self.kernel_size = kernel_size
 227          self.dilation_rate = dilation_rate
 228          self.n_layers = n_layers
 229          self.n_flows = n_flows
 230          self.gin_channels = gin_channels
 231  
 232          self.flows = nn.ModuleList()
 233          for i in range(n_flows):
 234              self.flows.append(
 235                  modules.ResidualCouplingLayer(
 236                      channels,
 237                      hidden_channels,
 238                      kernel_size,
 239                      dilation_rate,
 240                      n_layers,
 241                      gin_channels=gin_channels,
 242                      mean_only=True,
 243                  )
 244              )
 245              self.flows.append(modules.Flip())
 246  
 247      def forward(self, x, x_mask, g=None, reverse=False):
 248          if not reverse:
 249              for flow in self.flows:
 250                  x, _ = flow(x, x_mask, g=g, reverse=reverse)
 251          else:
 252              for flow in reversed(self.flows):
 253                  x = flow(x, x_mask, g=g, reverse=reverse)
 254          return x
 255  
 256  
```

### lines 299–376
```python
 299  class Generator(torch.nn.Module):
 300      def __init__(
 301          self,
 302          initial_channel: int,
 303          resblock: typing.Optional[str],
 304          resblock_kernel_sizes: typing.Tuple[int, ...],
 305          resblock_dilation_sizes: typing.Tuple[typing.Tuple[int, ...], ...],
 306          upsample_rates: typing.Tuple[int, ...],
 307          upsample_initial_channel: int,
 308          upsample_kernel_sizes: typing.Tuple[int, ...],
 309          gin_channels: int = 0,
 310      ):
 311          super(Generator, self).__init__()
 312          self.LRELU_SLOPE = 0.1
 313          self.num_kernels = len(resblock_kernel_sizes)
 314          self.num_upsamples = len(upsample_rates)
 315          self.conv_pre = Conv1d(
 316              initial_channel, upsample_initial_channel, 7, 1, padding=3
 317          )
 318          resblock_module = modules.ResBlock1 if resblock == "1" else modules.ResBlock2
 319  
 320          self.ups = nn.ModuleList()
 321          for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
 322              self.ups.append(
 323                  weight_norm(
 324                      ConvTranspose1d(
 325                          upsample_initial_channel // (2**i),
 326                          upsample_initial_channel // (2 ** (i + 1)),
 327                          k,
 328                          u,
 329                          padding=(k - u) // 2,
 330                      )
 331                  )
 332              )
 333  
 334          self.resblocks = nn.ModuleList()
 335          for i in range(len(self.ups)):
 336              ch = upsample_initial_channel // (2 ** (i + 1))
 337              for j, (k, d) in enumerate(
 338                  zip(resblock_kernel_sizes, resblock_dilation_sizes)
 339              ):
 340                  self.resblocks.append(resblock_module(ch, k, d))
 341  
 342          self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
 343          self.ups.apply(init_weights)
 344  
 345          if gin_channels != 0:
 346              self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)
 347  
 348      def forward(self, x, g=None):
 349          x = self.conv_pre(x)
 350          if g is not None:
 351              x = x + self.cond(g)
 352  
 353          for i, up in enumerate(self.ups):
 354              x = F.leaky_relu(x, self.LRELU_SLOPE)
 355              x = up(x)
 356              xs = torch.zeros(1)
 357              for j, resblock in enumerate(self.resblocks):
 358                  index = j - (i * self.num_kernels)
 359                  if index == 0:
 360                      xs = resblock(x)
 361                  elif (index > 0) and (index < self.num_kernels):
 362                      xs += resblock(x)
 363              x = xs / self.num_kernels
 364          x = F.leaky_relu(x)
 365          x = self.conv_post(x)
 366          x = torch.tanh(x)
 367  
 368          return x
 369  
 370      def remove_weight_norm(self):
 371          print("Removing weight norm...")
 372          for l in self.ups:
 373              remove_weight_norm(l)
 374          for l in self.resblocks:
 375              l.remove_weight_norm()
 376  
```

### lines 630–655
```python
 630              s_p_sq_r = torch.exp(-2 * logs_p)  # [b, d, t]
 631              neg_cent1 = torch.sum(
 632                  -0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True
 633              )  # [b, 1, t_s]
 634              neg_cent2 = torch.matmul(
 635                  -0.5 * (z_p**2).transpose(1, 2), s_p_sq_r
 636              )  # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
 637              neg_cent3 = torch.matmul(
 638                  z_p.transpose(1, 2), (m_p * s_p_sq_r)
 639              )  # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
 640              neg_cent4 = torch.sum(
 641                  -0.5 * (m_p**2) * s_p_sq_r, [1], keepdim=True
 642              )  # [b, 1, t_s]
 643              neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4
 644  
 645              attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
 646              attn = (
 647                  monotonic_align.maximum_path(neg_cent, attn_mask.squeeze(1))
 648                  .unsqueeze(1)
 649                  .detach()
 650              )
 651  
 652          w = attn.sum(2)
 653          if self.use_sdp:
 654              l_length = self.dp(x, x_mask, w, g=g)
 655              l_length = l_length / torch.sum(x_mask)
```

## piper:modules.py
SHA-256 `c3a32fdf18c62b7048f3947b7d04416f791d1e406088c43f776ae1ea7396fda4`, 527 lines

### lines 220–240
```python
 220  class ResBlock1(torch.nn.Module):
 221      def __init__(
 222          self,
 223          channels: int,
 224          kernel_size: int = 3,
 225          dilation: typing.Tuple[int] = (1, 3, 5),
 226      ):
 227          super(ResBlock1, self).__init__()
 228          self.LRELU_SLOPE = 0.1
 229          self.convs1 = nn.ModuleList(
 230              [
 231                  weight_norm(
 232                      Conv1d(
 233                          channels,
 234                          channels,
 235                          kernel_size,
 236                          1,
 237                          dilation=dilation[0],
 238                          padding=get_padding(kernel_size, dilation[0]),
 239                      )
 240                  ),
```

### lines 323–380
```python
 323  class ResBlock2(torch.nn.Module):
 324      def __init__(
 325          self, channels: int, kernel_size: int = 3, dilation: typing.Tuple[int] = (1, 3)
 326      ):
 327          super(ResBlock2, self).__init__()
 328          self.LRELU_SLOPE = 0.1
 329          self.convs = nn.ModuleList(
 330              [
 331                  weight_norm(
 332                      Conv1d(
 333                          channels,
 334                          channels,
 335                          kernel_size,
 336                          1,
 337                          dilation=dilation[0],
 338                          padding=get_padding(kernel_size, dilation[0]),
 339                      )
 340                  ),
 341                  weight_norm(
 342                      Conv1d(
 343                          channels,
 344                          channels,
 345                          kernel_size,
 346                          1,
 347                          dilation=dilation[1],
 348                          padding=get_padding(kernel_size, dilation[1]),
 349                      )
 350                  ),
 351              ]
 352          )
 353          self.convs.apply(init_weights)
 354  
 355      def forward(self, x, x_mask=None):
 356          for c in self.convs:
 357              xt = F.leaky_relu(x, self.LRELU_SLOPE)
 358              if x_mask is not None:
 359                  xt = xt * x_mask
 360              xt = c(xt)
 361              x = xt + x
 362          if x_mask is not None:
 363              x = x * x_mask
 364          return x
 365  
 366      def remove_weight_norm(self):
 367          for l in self.convs:
 368              remove_weight_norm(l)
 369  
 370  
 371  class Log(nn.Module):
 372      def forward(
 373          self, x: torch.Tensor, x_mask: torch.Tensor, reverse: bool = False, **kwargs
 374      ):
 375          if not reverse:
 376              y = torch.log(torch.clamp_min(x, 1e-5)) * x_mask
 377              logdet = torch.sum(-y, [1, 2])
 378              return y, logdet
 379          else:
 380              x = torch.exp(x) * x_mask
```

### lines 412–470
```python
 412  class ResidualCouplingLayer(nn.Module):
 413      def __init__(
 414          self,
 415          channels: int,
 416          hidden_channels: int,
 417          kernel_size: int,
 418          dilation_rate: int,
 419          n_layers: int,
 420          p_dropout: float = 0,
 421          gin_channels: int = 0,
 422          mean_only: bool = False,
 423      ):
 424          assert channels % 2 == 0, "channels should be divisible by 2"
 425          super().__init__()
 426          self.channels = channels
 427          self.hidden_channels = hidden_channels
 428          self.kernel_size = kernel_size
 429          self.dilation_rate = dilation_rate
 430          self.n_layers = n_layers
 431          self.half_channels = channels // 2
 432          self.mean_only = mean_only
 433  
 434          self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
 435          self.enc = WN(
 436              hidden_channels,
 437              kernel_size,
 438              dilation_rate,
 439              n_layers,
 440              p_dropout=p_dropout,
 441              gin_channels=gin_channels,
 442          )
 443          self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
 444          self.post.weight.data.zero_()
 445          self.post.bias.data.zero_()
 446  
 447      def forward(self, x, x_mask, g=None, reverse=False):
 448          x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
 449          h = self.pre(x0) * x_mask
 450          h = self.enc(h, x_mask, g=g)
 451          stats = self.post(h) * x_mask
 452          if not self.mean_only:
 453              m, logs = torch.split(stats, [self.half_channels] * 2, 1)
 454          else:
 455              m = stats
 456              logs = torch.zeros_like(m)
 457  
 458          if not reverse:
 459              x1 = m + x1 * torch.exp(logs) * x_mask
 460              x = torch.cat([x0, x1], 1)
 461              logdet = torch.sum(logs, [1, 2])
 462              return x, logdet
 463          else:
 464              x1 = (x1 - m) * torch.exp(-logs) * x_mask
 465              x = torch.cat([x0, x1], 1)
 466              return x
 467  
 468  
 469  class ConvFlow(nn.Module):
 470      def __init__(
```

