"""The pure functional tick: no object state, no in-place mutation of inputs, so it can
be wrapped in torch.compile once and reused for both the no-grad and autograd paths."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

# (dy, dx) for the 8-neighbourhood, orthogonals first.
NEIGHBOURS = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))
MOVES = ((0, 0),) + NEIGHBOURS


@dataclass(frozen=True)
class Params:
    K_RELAX: int
    acts_per_tick: int
    temp: float
    tau: float
    soft_field: bool
    SPEED: float
    RATE_DT: float
    DEFEND_DT: float
    REGEN_DT: float
    threshold_sharpness: float
    attack_isotropy: float
    track_health: bool
    new_health: float
    MAX_CURSOR_SPEED: float
    action_mode: str
    capacity: float
    crowd_sharpness: float
    capacity_iters: int
    seed_radius: float
    LARGE: float
    diag_cost: float

    def __post_init__(self):
        # SPEED is the fraction of a cell's mass that leaves it each tick. Above 1 the
        # cell keeps density * (1 - SPEED) < 0, densities go negative and the whole
        # sim NaNs a few ticks later with no other warning.
        if not 0.0 <= self.SPEED <= 1.0:
            raise ValueError(f"SPEED must be in [0, 1], got {self.SPEED}")
        if not 0.0 <= self.RATE_DT <= 1.0:
            raise ValueError(f"RATE_DT must be in [0, 1] to keep densities non-negative, "
                             f"got {self.RATE_DT}")

    @staticmethod
    def from_cfg(cfg) -> "Params":
        s = cfg.sim
        return Params(
            K_RELAX=int(s.K_RELAX),
            acts_per_tick=int(s.get("acts_per_tick", 1)), temp=float(s.temp), tau=float(s.tau),
            soft_field=bool(s.soft_field), SPEED=float(s.SPEED),
            RATE_DT=float(s.RATE_DT),
            DEFEND_DT=float(s.get("DEFEND_DT", 0.005)),
            REGEN_DT=float(s.get("REGEN_DT", 0.0005)),
            threshold_sharpness=float(s.get("threshold_sharpness", 3.0)),
            attack_isotropy=float(s.get("attack_isotropy", 0.0)),
            track_health=bool(s.get("track_health", False)),
            new_health=float(s.get("new_health", 0.5)),
            MAX_CURSOR_SPEED=float(s.MAX_CURSOR_SPEED),
            action_mode=str(s.get("action_mode", "velocity")),
            capacity=float(s.get("capacity", 1.0)),
            crowd_sharpness=float(s.get("crowd_sharpness", 3.0)),
            capacity_iters=int(s.get("capacity_iters", 2)),
            seed_radius=float(s.seed_radius),
            LARGE=float(s.LARGE), diag_cost=float(s.diag_cost),
        )


def _pad(t: torch.Tensor, fill: float) -> torch.Tensor:
    return F.pad(t, (1, 1, 1, 1), mode="constant", value=fill)


def _nbr(padded: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    """View of the neighbour at offset (dy, dx) of every cell. No allocation."""
    H = padded.shape[-2] - 2
    W = padded.shape[-1] - 2
    return padded[..., 1 + dy : 1 + dy + H, 1 + dx : 1 + dx + W]


def _scatter(padded_flow: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    """The mass that cell x sent along (dy, dx), viewed at its destination x + (dy, dx)."""
    return _nbr(padded_flow, -dy, -dx)


def cursor_seed(cursor, walls, radius: float, large: float):
    """Differentiable seed: euclidean distance inside `radius`, LARGE outside.

    A hard one-hot index at the cursor cell has zero gradient w.r.t. the cursor, which
    would silently kill the whole method. Euclidean == geodesic in a small open
    neighbourhood, so this is exact as well as differentiable.
    """
    B, T, _ = cursor.shape
    H, W = walls.shape[-2:]
    ys = torch.arange(H, device=cursor.device, dtype=cursor.dtype).view(1, 1, H, 1)
    xs = torch.arange(W, device=cursor.device, dtype=cursor.dtype).view(1, 1, 1, W)
    dy = ys - cursor[..., 0].view(B, T, 1, 1)
    dx = xs - cursor[..., 1].view(B, T, 1, 1)
    dist = torch.sqrt(dy * dy + dx * dx + 1e-8)
    inside = torch.sigmoid((radius - dist) * 4.0)
    seed = dist * inside + large * (1.0 - inside)
    op = walls.unsqueeze(1)
    return seed * op + large * (1.0 - op)


def relax(pot, seed, openness, *, k: int, p: Params):
    """Bellman value iteration for the geodesic field.

        pot'(x) = min( seed(x), min_d pot(x + d) + cost_d )

    `pot` itself is deliberately not a candidate: with it the operator is monotonically
    decreasing, so the well dug by the cursor's previous position would never fill back
    in and the army would keep flowing to where the cursor used to be. Without it,
    values rise as well as fall and the iteration converges to the true geodesic
    distance -- while still being cheap enough to run only K_RELAX passes per tick, so
    the field lags the cursor slightly the way the original game's does.
    """
    large, dc = p.LARGE, p.diag_cost
    wall_pot = large * (1.0 - openness)
    for _ in range(k):
        pp = _pad(pot, large)
        if p.soft_field:
            cands = [seed]
            for dy, dx in NEIGHBOURS:
                cands.append(_nbr(pp, dy, dx) + (1.0 if dy == 0 or dx == 0 else dc))
            pot = -p.tau * torch.logsumexp(-torch.stack(cands, 0) / p.tau, dim=0)
        else:
            cross = torch.minimum(
                torch.minimum(_nbr(pp, -1, 0), _nbr(pp, 1, 0)),
                torch.minimum(_nbr(pp, 0, -1), _nbr(pp, 0, 1)),
            )
            diag = torch.minimum(
                torch.minimum(_nbr(pp, -1, -1), _nbr(pp, -1, 1)),
                torch.minimum(_nbr(pp, 1, -1), _nbr(pp, 1, 1)),
            )
            pot = torch.minimum(seed, torch.minimum(cross + 1.0, diag + dc))
        pot = pot.clamp(max=large) * openness + wall_pot
    return pot


def advect(density, pot, openness, p: Params, stock=None):
    """Mass-conserving flow down the potential gradient over the 9 moves.

    Returns (new density, attack pressure). The attack pressure at a cell is the mass
    of each team that tried to move into that cell and was refused: in LW6 a fighter
    fights only when it cannot move, and it swings in the direction its cursor is
    pulling it. `attack_isotropy` blends in a component that spreads a cell's blocked
    mass over all eight neighbours instead, because a purely directional rule makes a
    defender facing its own cursor unable to fight back at all.

    With `capacity > 0` this is a fixed-point iteration, and both halves of it matter:

    * A destination cannot hold more than `capacity`. Merely discounting a full
      destination's attractiveness cannot refuse anything -- the weights are
      renormalised, so if a cell and all its neighbours are equally full the potential
      gradient wins and mass piles in anyway. Measured: an army of 300 collapsing onto
      two cells at density 150 each, mass perfectly conserved all the way down.

    * Room has to be measured against what a cell will *actually* hold, and mass that a
      full neighbour refuses bounces back and still occupies the cell it came from.
      Measure room against only the mass intending to stay and the cell at the bottom
      of the well looks empty -- it is trying to leave in every direction and being
      refused in every direction -- so it accepts a full capacity of new mass every
      tick. Measured: densities climbing ~0.94 per tick without limit.

    * Refused mass is re-offered to the other directions rather than just staying put,
      by folding the acceptance back into the move weights. That is LW6's move fan: a
      blocked fighter tries the next direction round before giving up. Without it an
      army flows as a one-cell filament along a single geodesic instead of advancing
      as a body.

    Refusal propagates one cell backwards per round, so a few rounds settle it.
    """
    pp = _pad(pot, p.LARGE)
    po = _pad(openness, 0.0)

    # Potential drops in log space. Subtracting the max over *reachable* moves is what
    # keeps this finite: take the max over the raw drops instead and the remaining
    # exponents underflow to exactly 0 at low `temp` whenever the steepest move happens
    # to be into a wall, the renormalisation then divides by ~0, and the cell's mass
    # silently disappears -- 1.8% of total mass over 2000 ticks.
    SENTINEL = -1.0e30
    drops, opens, m = [], [], None
    for dy, dx in MOVES:
        d = (pot - _nbr(pp, dy, dx)) / p.temp
        o = _nbr(po, dy, dx)
        drops.append(d)
        opens.append(o)
        cand = torch.where(o > 0, d, torch.full_like(d, SENTINEL))
        m = cand if m is None else torch.maximum(m, cand)
    m = m.detach()  # a constant shift leaves the softmax unchanged

    def allocate(accept):
        """Split each cell's moving mass over the 9 moves, given who will accept it.

        Done in log space and renormalised by the best *available* move, so the
        normaliser is always >= 1. Doing it on raw weights instead lets every term
        underflow to exactly 0 in float32 -- at temp = 0.1 an 8-unit potential drop is
        already exp(-80) -- and the division then destroys the cell's mass. Measured at
        1.8% of total mass lost over 2000 ticks, twice, by two different routes.
        """
        pa = None if accept is None else _pad(accept, 1.0)
        scores, best = [], None
        for i, (dy, dx) in enumerate(MOVES):
            sc = drops[i] - m
            if pa is not None and i > 0:
                sc = sc + torch.log(_nbr(pa, dy, dx).clamp_min(1e-30))
            scores.append(sc)
            cand = torch.where(opens[i] > 0, sc, torch.full_like(sc, SENTINEL))
            best = cand if best is None else torch.maximum(best, cand)
        best = best.detach()
        # max=0: `best` is the max over reachable moves, so a reachable exponent is
        # never positive -- but a wall cell has no reachable move and its `best` is
        # the sentinel, which without the upper clamp overflows to inf and makes the
        # backward pass return NaN.
        ws = [torch.exp((sc - best).clamp(min=-80.0, max=0.0)) * o
              for sc, o in zip(scores, opens)]
        tot = ws[0]
        for w in ws[1:]:
            tot = tot + w
        # Additive, not clamp_min: a wall cell has tot == 0, and 1/clamp_min(0, 1e-20)
        # has a backward of -1e40, which overflows to inf and poisons the gradient with
        # NaN even though the forward value is fine. For any open cell tot >= 1.
        # The split fractions, not the split mass: with a health field the same
        # fractions have to move the health stock as well as the mass, or health
        # teleports relative to the fighters carrying it.
        inv = p.SPEED / (tot + 1e-9)
        return [w * inv for w in ws]

    def transport(frac, field, accept):
        """Move `field` with the given per-move fractions and acceptance."""
        desired = [field * f for f in frac]
        out = field * (1.0 - p.SPEED) + desired[0]
        if accept is None:
            for i, (dy, dx) in enumerate(MOVES[1:], start=1):
                out = out + _scatter(_pad(desired[i], 0.0), dy, dx)
            return out, desired
        pa_ = _pad(accept, 1.0)
        for i, (dy, dx) in enumerate(MOVES[1:], start=1):
            arriving = _scatter(_pad(desired[i], 0.0), dy, dx)
            out = out + arriving * accept + desired[i] * (1.0 - _nbr(pa_, dy, dx))
        return out, desired

    if p.capacity <= 0:
        frac = allocate(None)
        out, _ = transport(frac, density, None)
        out_stock = None
        if stock is not None:
            out_stock, _ = transport(frac, stock, None)
        return out * openness, torch.zeros_like(out), \
            (out_stock * openness if out_stock is not None else None)

    beta = p.crowd_sharpness
    accept, frac = None, None
    for _ in range(max(int(p.capacity_iters), 1)):
        frac = allocate(accept)
        inflow, gone = None, None
        pa = None if accept is None else _pad(accept, 1.0)
        for i, (dy, dx) in enumerate(MOVES[1:], start=1):
            sent = density * frac[i]
            arriving = _scatter(_pad(sent, 0.0), dy, dx)
            inflow = arriving if inflow is None else inflow + arriving
            g = sent if pa is None else sent * _nbr(pa, dy, dx)
            gone = g if gone is None else gone + g
        occupied = (density - gone).sum(1, keepdim=True)
        # softplus, not relu: relu(capacity - occupied) has a kink exactly at full,
        # which is where the gradient matters most. beta trades a little slack above
        # capacity for smoothness.
        room = F.softplus((p.capacity - occupied) * beta) / beta
        # 1 - exp(-room/want) is smooth, tends to 1 when there is plenty of room, and
        # satisfies accept * want <= room, so the cap is never exceeded.
        accept = 1.0 - torch.exp(-(room / (inflow.sum(1, keepdim=True) + 1e-9)).clamp(max=60.0))

    frac = allocate(accept)
    out, _ = transport(frac, density, accept)
    out_stock = None
    if stock is not None:
        # The health stock rides the same fractions and the same acceptance, so
        # health stays attached to the fighters carrying it.
        out_stock, _ = transport(frac, stock, accept)

    # Attack pressure, and it has to be *directional*.
    #
    # A LW6 fighter swings in the direction its cursor is pulling it, at whatever is
    # in the way. So an army that has enveloped another attacks inward, while the
    # enveloped army's own fan also points inward -- at its own allies, whom it heals
    # instead of fighting. That asymmetry is the whole reason charging in and
    # surrounding is strong in this game. An isotropic 8-neighbour pressure sum cannot
    # express it: with one, the density sim had the *defender* winning a matchup the
    # real kernel decides the other way, ~90% curve error however the free parameters
    # were tuned.
    #
    # So: mass that wants to move in direction d and is refused delivers its attack to
    # the cell in direction d. The preference is the unconstrained one -- a LW6 fighter
    # tries only the forward directions of its fan and attacks if those are taken, even
    # though a cell behind it is free, so refusal must be measured over where the mass
    # wanted to go, not over everywhere it could have gone.
    pa = _pad(accept, 1.0)
    prefs, ptot = [], None
    for i in range(1, len(MOVES)):
        pr = torch.exp((drops[i] - m).clamp(min=-80.0, max=0.0)) * opens[i]
        prefs.append(pr)
        ptot = pr if ptot is None else ptot + pr
    attack, blocked_total = None, None
    for i, (dy, dx) in enumerate(MOVES[1:], start=1):
        flux = density * (prefs[i - 1] / (ptot + 1e-9)) * (1.0 - _nbr(pa, dy, dx))
        blocked_total = flux if blocked_total is None else blocked_total + flux
        arriving = _scatter(_pad(flux, 0.0), dy, dx)
        attack = arriving if attack is None else attack + arriving

    # Purely directional attack is too sharp an asymmetry. A LW6 fighter tries
    # `nb_attack_tries` directions of its fan, not one, and a packed blob's interior
    # fighters are blocked in every direction at once -- so a defender facing its own
    # cursor still hits the enemy behind it. With no isotropic component at all the
    # defender never fights back: the density sim handed the attacker 1.00 of the
    # population in a matchup the real kernel scored 0.43.
    if p.attack_isotropy > 0.0:
        iso = None
        share = blocked_total / 8.0
        for dy, dx in NEIGHBOURS:
            arriving = _scatter(_pad(share, 0.0), dy, dx)
            iso = arriving if iso is None else iso + arriving
        attack = (1.0 - p.attack_isotropy) * attack + p.attack_isotropy * iso
    return out * openness, attack * openness, \
        (out_stock * openness if out_stock is not None else None)


def combat(density, attack, p: Params, eps: float = 1e-9):
    """Conversion, as the mean-field limit of the LW6 attack rule (no health field).

    `attack[:, t]` is how much of team t's mass tried to move into each cell and was
    refused. In LW6 a fighter fights only when it cannot move, and it swings in the
    direction its cursor is pulling it.

    A blocked fighter attacks what is in the way if it is an enemy and *heals* it if
    it is an ally, and one that does neither regenerates -- so the same pressure field
    is damage when it lands on the other team and defence when it lands on its own,
    and a cell only starts losing mass once the damage beats the healing. All three
    constants come from LW6's rule defaults over `max_fighter_health`, so none is free.

    Softplus is deliberately not used for the threshold: it has a floor of log(2)/k at
    zero, so with no enemy anywhere the armies still converted each other. The
    multiplicative form below is exactly zero whenever the damage is.
    """
    a, b = density[:, 0], density[:, 1]
    pa, pb = attack[:, 0], attack[:, 1]
    k = p.threshold_sharpness

    def convert(defender, enemy_pressure, friendly_pressure):
        dmg = p.RATE_DT * enemy_pressure
        heal = p.DEFEND_DT * friendly_pressure + p.REGEN_DT * defender
        net = dmg * (dmg / (dmg + heal + eps)) ** k
        return defender * (1.0 - torch.exp(-(net / (defender + eps)).clamp(max=60.0)))

    net = convert(b, pa, pb) - convert(a, pb, pa)
    return torch.stack([a + net, b - net], dim=1)


def combat_with_health(density, stock, attack, p: Params, eps: float = 1e-9):
    """Conversion with an explicit health field. Returns (density, stock).

    `stock[:, t]` is team t's health stock in a cell -- mass times mean health
    fraction -- so `h = stock / mass`. Carrying it fixes the model's largest
    remaining error. Without health, a conversion is permanent the instant it
    happens, so a local advantage compounds; in LW6 a converted fighter respawns at
    `fighter_new_health` (half) deep inside the enemy blob and is usually converted
    straight back, and that churn is what makes charging a packed defender expensive.

    Within a cell, health is taken to be uniform on [0, 2h]. Damage `d` per unit mass
    lowers everyone by `d`, so the fraction below zero -- the fraction that flips --
    is `d / 2h`, smoothed to `1 - exp(-d / 2h)` so it saturates at 1 and stays
    differentiable. Survivors' mean health drops by `d / 2`.
    """
    m = density
    h = (stock / (m + eps)).clamp(0.0, 1.0)
    pressure = attack
    enemy_pressure = pressure.flip(1)          # what the other team is pressing in
    friendly_pressure = pressure

    dmg = p.RATE_DT * enemy_pressure
    heal = p.DEFEND_DT * friendly_pressure + p.REGEN_DT * m
    # Net health removed per unit of mass this tick.
    d = ((dmg - heal) / (m + eps)).clamp(min=0.0)

    flip_frac = 1.0 - torch.exp(-(d / (2.0 * h + eps)).clamp(max=60.0))
    converted = m * flip_frac                  # [B, 2, H, W]: mass leaving each team
    survivors = m - converted
    surv_h = (h - 0.5 * d).clamp(0.0, 1.0)

    gained = converted.flip(1)                 # what each team takes from the other
    new_m = survivors + gained
    new_stock = survivors * surv_h + gained * p.new_health
    # Healing also tops survivors up, bounded by full health.
    new_stock = torch.minimum(new_stock, new_m)
    return new_m, new_stock


def move_cursor(cursor, action, walls, p: Params):
    """Place each cursor for this round.

    Two action spaces. `target` takes a normalised position in [0, 1] and puts the
    cursor there -- what LW6's own bots do, and a strictly larger action space.
    `velocity` takes a direction and moves at most MAX_CURSOR_SPEED cells, which is
    what a hand can do.

    Neither blocks on walls: LW6 applies a cursor that lands on a wall at the nearest
    free slot instead, and refusing the move wedges a straight-line chaser in the
    first concave corner for the rest of the episode.
    """
    H, W = walls.shape[-2:]
    if p.action_mode == "target":
        scale = torch.tensor([H - 1.0, W - 1.0], device=cursor.device,
                             dtype=cursor.dtype)
        return action.clamp(0.0, 1.0) * scale
    return _move_cursor_velocity(cursor, action, walls, p)


def _move_cursor_velocity(cursor, action, walls, p: Params):
    """Move each cursor, sliding along walls rather than sticking to them.

    A cursor that simply refuses a blocked move gets wedged against the first wall it
    meets and stays there for the rest of the episode -- which silently turns most
    games into no-contact draws. Trying the full move, then the y-only and x-only
    components, is the usual collision slide and keeps every agent (scripted or
    learned) able to get around geometry.
    """
    H, W = walls.shape[-2:]
    B, T = cursor.shape[0], cursor.shape[1]
    bb = torch.arange(B, device=cursor.device).view(B, 1).expand(B, T)

    def clamp(c):
        return torch.stack([c[..., 0].clamp(1.0, H - 2.0), c[..., 1].clamp(1.0, W - 2.0)], -1)

    def is_open(c):
        idx = c.detach().round().long()
        return walls[bb, idx[..., 0], idx[..., 1]] > 0.5

    step = action * p.MAX_CURSOR_SPEED
    full = clamp(cursor + step)
    y_only = clamp(cursor + step * torch.tensor([1.0, 0.0], device=cursor.device))
    x_only = clamp(cursor + step * torch.tensor([0.0, 1.0], device=cursor.device))

    out = cursor
    for cand in (x_only, y_only, full):  # later candidates win, so full move is preferred
        out = torch.where(is_open(cand).unsqueeze(-1), cand, out)
    return out


def score(density):
    return density[:, 0].sum(dim=(1, 2)) / density.sum(dim=(1, 2, 3)).clamp_min(1e-12)


def tick(density, pot, cursor, walls, action, p: Params, stock=None):
    """One LW6 round: move the cursor, spread the gradient, then act `acts_per_tick`
    times against that one gradient. LW6's default is 2 moves and 5 spreads per round,
    so a fighter covers two cells per round while the field is updated once.

    With `track_health`, `stock` carries each team's health stock and is returned
    alongside the density; without it `stock` is None throughout.
    """
    openness = walls.unsqueeze(1)
    cursor = move_cursor(cursor, action, walls, p)
    seed = cursor_seed(cursor, walls, p.seed_radius, p.LARGE)
    pot = relax(pot, seed, openness, k=p.K_RELAX, p=p)
    for _ in range(max(p.acts_per_tick, 1)):
        density, attack, stock = advect(density, pot, openness, p, stock=stock)
        if stock is not None:
            density, stock = combat_with_health(density, stock, attack, p)
        else:
            density = combat(density, attack, p)
    return (density, pot, cursor) if stock is None else (density, pot, cursor, stock)
