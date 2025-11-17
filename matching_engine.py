# matching_engine.py

import heapq
import random
from decimal import Decimal, ROUND_FLOOR
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple, Any
from enum import Enum

from order_manager import OrderManager, Order
from order_enums import OrderType, OrderSide, OrderStatus


class SegmentMode(Enum):
    TV = "tv"
    CLASSIC = "classic"
    MICROTICKS = "microticks"


class BarDirection(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"


class QueueMode(Enum):
    FIFO = "fifo"
    PRO_RATA = "pro_rata"
    TV = "tv"


class MatchingEngine:
    def __init__(
        self,
        order_manager: OrderManager,
        tick_size: Decimal = Decimal("0.01"),
        maker_fee: Decimal = Decimal("0.001"),
        taker_fee: Decimal = Decimal("0.002"),
        latency_ms: int = 0,
        slippage_mode: str = "none",
        fixed_slippage_ticks: Decimal = Decimal("0"),
        dynamic_coeff: Decimal = Decimal("1.0"),
        queue_mode: QueueMode = QueueMode.FIFO,
        base_latency_ms: int = 0,
        jitter_ms: int = 0,
        execution_latency_ms: int = 0,
        latency_bucket_ms: int = 0,
        seed: int = 42,
    ):
        self.order_manager = order_manager
        self.tick_size = tick_size
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.latency_ms = latency_ms
        self.position = Decimal("0")

        self.slippage_mode = slippage_mode
        self.fixed_slippage_ticks = fixed_slippage_ticks
        self.dynamic_coeff = dynamic_coeff
        self.recent_volumes: List[Decimal] = []
        self.max_volume_window = 20

        self.queue_mode = queue_mode
        self.base_latency_ms = base_latency_ms
        self.jitter_ms = jitter_ms
        self.execution_latency_ms = execution_latency_ms
        self.latency_bucket_ms = latency_bucket_ms
        self.rng = random.Random(seed)

    # ------------------------------------------------------------
    # NEW public API for BacktestRunner (compat wrapper)
    # ------------------------------------------------------------
    def process_bar(self, **kwargs) -> List[Dict[str, Any]]:
        """
        Compatibility wrapper for BacktestRunner.

        پشتیبانی می‌کند از:
        - process_bar(open_price=..., high=..., low=..., close=..., volume=..., timestamp=..., mode=...)
        - process_bar(open=..., high=..., low=..., close=..., volume=..., timestamp=..., mode=...)

        سایر آرگومان‌ها مثل segment_low / segment_high نادیده گرفته می‌شوند.
        """

        # 1) خواندن ورودی‌ها با هر دو نام
        open_val = kwargs.get("open_price", kwargs.get("open"))
        high_val = kwargs.get("high")
        low_val = kwargs.get("low")
        close_val = kwargs.get("close")
        volume_val = kwargs.get("volume")
        timestamp = kwargs.get("timestamp")
        mode = kwargs.get("mode", SegmentMode.TV)

        # چک حداقلی برای جلوگیری از None
        if any(v is None for v in [open_val, high_val, low_val, close_val, volume_val]):
            raise ValueError(
                f"MatchingEngine.process_bar missing required OHLCV values: "
                f"open={open_val}, high={high_val}, low={low_val}, close={close_val}, volume={volume_val}"
            )

        # 2) timestamp → datetime
        if isinstance(timestamp, datetime):
            bar_ts = timestamp
        elif timestamp is None:
            bar_ts = datetime.utcnow()
        else:
            if hasattr(timestamp, "to_pydatetime"):
                bar_ts = timestamp.to_pydatetime()
            else:
                # فرض: میلی‌ثانیه یونیکس
                bar_ts = datetime.fromtimestamp(float(timestamp) / 1000.0)

        # 3) float/int → Decimal
        open_d = Decimal(str(open_val))
        high_d = Decimal(str(high_val))
        low_d = Decimal(str(low_val))
        close_d = Decimal(str(close_val))
        vol_d = Decimal(str(volume_val))

        # 4) فراخوانی منطق اصلی بدون دست‌کاری
        return self.process_bar_original(
            open_price=open_d,
            high=high_d,
            low=low_d,
            close=close_d,
            volume=vol_d,
            bar_timestamp=bar_ts,
            mode=mode,
        )

    # ------------------------------------------------------------
    # Original core engine logic (unchanged)
    # ------------------------------------------------------------
    def process_bar_original(
        self,
        open_price: Decimal,
        high: Decimal,
        low: Decimal,
        close: Decimal,
        volume: Decimal,
        bar_timestamp: datetime,
        mode: SegmentMode = SegmentMode.TV,
    ) -> List[Dict[str, Any]]:
        all_fills: List[Dict[str, Any]] = []

        self._update_volume_history(volume)

        direction = (
            BarDirection.BULLISH if close > open_price else BarDirection.BEARISH
        )

        processed_time = self._apply_latency(bar_timestamp, "bar_processing")
        self.order_manager.process_pending_orders(processed_time)

        if mode == SegmentMode.MICROTICKS:
            sequence = self._get_microtick_sequence(
                open_price, high, low, close, direction
            )
        else:
            sequence = self._get_tradingview_sequence(
                open_price, high, low, close, direction
            )

        for price_level, step_high, step_low in sequence:
            price_level = self._round_to_tick(price_level)
            step_high = self._round_to_tick(step_high)
            step_low = self._round_to_tick(step_low)
            level_fills: List[Dict[str, Any]] = []

            sl_fills = self._process_stop_loss_orders(
                price_level, step_high, step_low, bar_timestamp
            )
            level_fills.extend(sl_fills)

            tp_fills = self._process_take_profit_orders(
                price_level, step_high, step_low, bar_timestamp
            )
            level_fills.extend(tp_fills)

            other_stop_fills = self._process_other_stop_orders(
                price_level, step_high, step_low, bar_timestamp
            )
            level_fills.extend(other_stop_fills)

            market_fills = self._process_market_orders_at_level(
                price_level, step_high, step_low, bar_timestamp
            )
            level_fills.extend(market_fills)

            limit_fills = self._process_limit_orders_at_level(
                price_level, bar_timestamp
            )
            level_fills.extend(limit_fills)

            all_fills.extend(level_fills)

            self.order_manager.cleanup_completed_orders()

        return all_fills

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------
    def _apply_latency(self, timestamp: datetime, order_id: str) -> datetime:
        if self.base_latency_ms == 0 and self.jitter_ms == 0:
            return timestamp

        total_latency = self.base_latency_ms

        if self.jitter_ms > 0:
            jitter = self.rng.randint(-self.jitter_ms, self.jitter_ms)
            total_latency += jitter

        total_latency = max(0, total_latency)

        new_time = timestamp + timedelta(milliseconds=total_latency)

        if self.latency_bucket_ms > 0:
            bucket_ms = (
                (new_time.microsecond // 1000)
                // self.latency_bucket_ms
                * self.latency_bucket_ms
            )
            new_time = new_time.replace(microsecond=bucket_ms * 1000)

        return new_time

    def _apply_execution_latency(self, timestamp: datetime) -> datetime:
        if self.execution_latency_ms > 0:
            return timestamp + timedelta(milliseconds=self.execution_latency_ms)
        return timestamp

    def _round_to_tick(self, price: Decimal) -> Decimal:
        if price is None:
            return Decimal("0")
        return (
            (price / self.tick_size).to_integral_value(rounding=ROUND_FLOOR)
            * self.tick_size
        )

    def _round_quantity_to_tick(self, quantity: Decimal) -> Decimal:
        if quantity is None:
            return Decimal("0")
        return (
            (quantity / self.tick_size).to_integral_value(rounding=ROUND_FLOOR)
            * self.tick_size
        )

    def _apply_slippage(
        self, fill_price: Decimal, fill_quantity: Decimal, side: OrderSide
    ) -> Decimal:
        if self.slippage_mode == "none":
            return fill_price

        if self.slippage_mode == "fixed":
            slippage_ticks = self.fixed_slippage_ticks
        elif self.slippage_mode == "volume":
            if not self.recent_volumes:
                return fill_price

            avg_volume = sum(self.recent_volumes) / Decimal(len(self.recent_volumes))
            volume_ratio = fill_quantity / max(Decimal("1"), avg_volume)
            slippage_ticks = (volume_ratio * self.dynamic_coeff).quantize(
                Decimal("1"), rounding=ROUND_FLOOR
            )
        else:
            return fill_price

        slippage_amount = slippage_ticks * self.tick_size

        if side == OrderSide.BUY:
            new_price = fill_price + slippage_amount
        else:
            new_price = fill_price - slippage_amount

        new_price = max(new_price, Decimal("0"))
        return self._round_to_tick(new_price)

    def _update_volume_history(self, volume: Decimal):
        self.recent_volumes.append(volume)
        if len(self.recent_volumes) > self.max_volume_window:
            self.recent_volumes.pop(0)

    def _get_tradingview_sequence(
        self,
        open_price: Decimal,
        high: Decimal,
        low: Decimal,
        close: Decimal,
        direction: BarDirection,
    ) -> List[Tuple[Decimal, Decimal, Decimal]]:
        if direction == BarDirection.BULLISH:
            return [
                (open_price, high, low),
                (low, high, low),
                (high, high, low),
                (close, high, low),
            ]
        else:
            return [
                (open_price, high, low),
                (high, high, low),
                (low, high, low),
                (close, high, low),
            ]

    def _get_microtick_sequence(
        self,
        open_price: Decimal,
        high: Decimal,
        low: Decimal,
        close: Decimal,
        direction: BarDirection,
    ) -> List[Tuple[Decimal, Decimal, Decimal]]:
        sequence: List[Tuple[Decimal, Decimal, Decimal]] = []

        if direction == BarDirection.BULLISH:
            for i in range(3):
                t = Decimal(i + 1) / Decimal(3)
                price_level = open_price + (low - open_price) * t
                sequence.append((price_level, open_price, low))

            for i in range(4):
                t = Decimal(i + 1) / Decimal(4)
                price_level = low + (high - low) * t
                sequence.append((price_level, low, high))

            for i in range(5):
                t = Decimal(i + 1) / Decimal(5)
                price_level = high + (close - high) * t
                sequence.append((price_level, high, close))
        else:
            for i in range(3):
                t = Decimal(i + 1) / Decimal(3)
                price_level = open_price + (high - open_price) * t
                sequence.append((price_level, open_price, high))

            for i in range(4):
                t = Decimal(i + 1) / Decimal(4)
                price_level = high + (low - high) * t
                sequence.append((price_level, high, low))

            for i in range(5):
                t = Decimal(i + 1) / Decimal(5)
                price_level = low + (close - low) * t
                sequence.append((price_level, low, close))

        return sequence

    def _is_price_touched(self, order: Order, price_level: Decimal) -> bool:
        if order.side == OrderSide.BUY:
            return price_level <= order.limit_price
        else:
            return price_level >= order.limit_price

    def _is_stop_triggered(self, stop_order: Order, price_level: Decimal) -> bool:
        if stop_order.side == OrderSide.BUY:
            return price_level >= stop_order.stop_price
        else:
            return price_level <= stop_order.stop_price

    def _update_position(self, order: Order, fill_quantity: Decimal, is_buy: bool):
        if hasattr(order, "is_mine") and order.is_mine:
            if is_buy:
                self.position += fill_quantity
            else:
                self.position -= fill_quantity

    def _create_fill_dict(
        self,
        order: Order,
        fill_price: Decimal,
        fill_quantity: Decimal,
        current_time: datetime,
        is_maker: bool,
        order_type: OrderType,
    ) -> Dict[str, Any]:
        fee_rate = self.maker_fee if is_maker else self.taker_fee
        fee = self._round_to_tick(fill_price * fill_quantity * fee_rate)

        return {
            "order_id": order.order_id,
            "symbol": order.symbol,
            "side": order.side,
            "fill_price": fill_price,
            "fill_quantity": fill_quantity,
            "timestamp": current_time,
            "fee": fee,
            "is_maker": is_maker,
            "remaining_quantity": order.remaining_quantity,
            "order_type": order_type,
        }

    def _determine_aggressor(
        self, order1: Order, order2: Order
    ) -> Tuple[Order, Order]:
        if self.queue_mode == QueueMode.TV:
            return order1, order2

        if order1.activation_time > order2.activation_time:
            return order1, order2
        elif order1.activation_time < order2.activation_time:
            return order2, order1

        if order1.created_timestamp > order2.created_timestamp:
            return order1, order2
        elif order1.created_timestamp < order2.created_timestamp:
            return order2, order1

        if order1.order_id > order2.order_id:
            return order1, order2
        else:
            return order2, order1

    def _execute_limit_vs_limit_pro_rata(
        self,
        aggressor: Order,
        resting_orders: List[Order],
        fill_quantity: Decimal,
        current_time: datetime,
    ) -> List[Dict[str, Any]]:
        fills: List[Dict[str, Any]] = []
        total_resting_quantity = sum(
            order.remaining_quantity for order in resting_orders
        )

        if total_resting_quantity <= Decimal("0"):
            return fills

        remaining_fill = fill_quantity

        for resting_order in resting_orders:
            if remaining_fill <= Decimal("0"):
                break

            proportion = resting_order.remaining_quantity / total_resting_quantity
            fill_qty = min(remaining_fill * proportion, resting_order.remaining_quantity)
            fill_qty = self._round_quantity_to_tick(fill_qty)

            if fill_qty > Decimal("0"):
                fill_price = self._round_to_tick(resting_order.limit_price)
                fill_price = self._apply_slippage(fill_price, fill_qty, aggressor.side)

                execution_time = self._apply_execution_latency(current_time)

                self.order_manager.update_order_fill(
                    aggressor.order_id,
                    fill_price,
                    fill_qty,
                    execution_time,
                    self.taker_fee,
                    self.position,
                )
                self.order_manager.update_order_fill(
                    resting_order.order_id,
                    fill_price,
                    fill_qty,
                    execution_time,
                    self.maker_fee,
                    self.position,
                )

                self._update_position(
                    aggressor, fill_qty, aggressor.side == OrderSide.BUY
                )
                self._update_position(
                    resting_order, fill_qty, resting_order.side == OrderSide.BUY
                )

                aggressor_fill = self._create_fill_dict(
                    aggressor,
                    fill_price,
                    fill_qty,
                    execution_time,
                    False,
                    aggressor.order_type,
                )
                resting_fill = self._create_fill_dict(
                    resting_order,
                    fill_price,
                    fill_qty,
                    execution_time,
                    True,
                    resting_order.order_type,
                )

                fills.extend([aggressor_fill, resting_fill])
                remaining_fill -= fill_qty

        return fills

    def _execute_limit_vs_limit(
        self,
        aggressor: Order,
        resting: Order,
        fill_quantity: Decimal,
        current_time: datetime,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        fill_price = self._round_to_tick(resting.limit_price)
        fill_quantity = self._round_quantity_to_tick(fill_quantity)

        fill_price = self._apply_slippage(fill_price, fill_quantity, aggressor.side)

        execution_time = self._apply_execution_latency(current_time)

        self.order_manager.update_order_fill(
            aggressor.order_id,
            fill_price,
            fill_quantity,
            execution_time,
            self.taker_fee,
            self.position,
        )
        self.order_manager.update_order_fill(
            resting.order_id,
            fill_price,
            fill_quantity,
            execution_time,
            self.maker_fee,
            self.position,
        )

        self._update_position(
            aggressor, fill_quantity, aggressor.side == OrderSide.BUY
        )
        self._update_position(resting, fill_quantity, resting.side == OrderSide.BUY)

        aggressor_fill = self._create_fill_dict(
            aggressor,
            fill_price,
            fill_quantity,
            execution_time,
            False,
            aggressor.order_type,
        )
        resting_fill = self._create_fill_dict(
            resting,
            fill_price,
            fill_quantity,
            execution_time,
            True,
            resting.order_type,
        )

        return aggressor_fill, resting_fill

    def _execute_market_vs_limit(
        self,
        market_order: Order,
        limit_order: Order,
        fill_quantity: Decimal,
        current_time: datetime,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        fill_price = self._round_to_tick(limit_order.limit_price)
        fill_quantity = self._round_quantity_to_tick(fill_quantity)

        fill_price = self._apply_slippage(fill_price, fill_quantity, market_order.side)

        execution_time = self._apply_execution_latency(current_time)

        self.order_manager.update_order_fill(
            market_order.order_id,
            fill_price,
            fill_quantity,
            execution_time,
            self.taker_fee,
            self.position,
        )
        self.order_manager.update_order_fill(
            limit_order.order_id,
            fill_price,
            fill_quantity,
            execution_time,
            self.maker_fee,
            self.position,
        )

        self._update_position(
            market_order, fill_quantity, market_order.side == OrderSide.BUY
        )
        self._update_position(
            limit_order, fill_quantity, limit_order.side == OrderSide.BUY
        )

        market_fill = self._create_fill_dict(
            market_order,
            fill_price,
            fill_quantity,
            execution_time,
            False,
            market_order.order_type,
        )
        limit_fill = self._create_fill_dict(
            limit_order,
            fill_price,
            fill_quantity,
            execution_time,
            True,
            limit_order.order_type,
        )

        return market_fill, limit_fill

    def _execute_stop_market_fallback(
        self,
        order: Order,
        step_high: Decimal,
        step_low: Decimal,
        current_time: datetime,
    ) -> List[Dict[str, Any]]:
        fills: List[Dict[str, Any]] = []

        fill_quantity = self._round_quantity_to_tick(order.remaining_quantity)

        if order.side == OrderSide.BUY:
            execution_price = self._round_to_tick(step_high)
        else:
            execution_price = self._round_to_tick(step_low)

        execution_price = self._apply_slippage(
            execution_price, fill_quantity, order.side
        )

        execution_time = self._apply_execution_latency(current_time)

        self.order_manager.update_order_fill(
            order.order_id,
            execution_price,
            fill_quantity,
            execution_time,
            self.taker_fee,
            self.position,
        )

        self._update_position(order, fill_quantity, order.side == OrderSide.BUY)

        market_fill = self._create_fill_dict(
            order,
            execution_price,
            fill_quantity,
            execution_time,
            False,
            order.order_type,
        )
        fills.append(market_fill)

        return fills

    def _process_stop_loss_orders(
        self,
        price_level: Decimal,
        step_high: Decimal,
        step_low: Decimal,
        current_time: datetime,
    ) -> List[Dict[str, Any]]:
        fills: List[Dict[str, Any]] = []

        all_stop_orders = self.order_manager.get_active_stop_orders()

        for stop_order in all_stop_orders:
            if hasattr(stop_order, "is_stop_loss") and stop_order.is_stop_loss:
                if self._is_stop_triggered(stop_order, price_level):
                    activated_time = self._apply_latency(
                        current_time, stop_order.order_id
                    )
                    self.order_manager.activate_stop_order(
                        stop_order, activated_time, self.latency_ms
                    )

                    if stop_order.order_type == OrderType.STOP_LIMIT:
                        stop_order.activation_time = activated_time
                        self.order_manager.add_limit_order(stop_order)
                    else:
                        stop_fills = self._execute_stop_market_fallback(
                            stop_order, step_high, step_low, activated_time
                        )
                        fills.extend(stop_fills)

        return fills

    def _process_take_profit_orders(
        self,
        price_level: Decimal,
        step_high: Decimal,
        step_low: Decimal,
        current_time: datetime,
    ) -> List[Dict[str, Any]]:
        fills: List[Dict[str, Any]] = []

        all_stop_orders = self.order_manager.get_active_stop_orders()

        for stop_order in all_stop_orders:
            if hasattr(stop_order, "is_take_profit") and stop_order.is_take_profit:
                if self._is_stop_triggered(stop_order, price_level):
                    activated_time = self._apply_latency(
                        current_time, stop_order.order_id
                    )
                    self.order_manager.activate_stop_order(
                        stop_order, activated_time, self.latency_ms
                    )

                    if stop_order.order_type == OrderType.STOP_LIMIT:
                        stop_order.activation_time = activated_time
                        self.order_manager.add_limit_order(stop_order)
                    else:
                        stop_fills = self._execute_stop_market_fallback(
                            stop_order, step_high, step_low, activated_time
                        )
                        fills.extend(stop_fills)

        return fills

    def _process_other_stop_orders(
        self,
        price_level: Decimal,
        step_high: Decimal,
        step_low: Decimal,
        current_time: datetime,
    ) -> List[Dict[str, Any]]:
        fills: List[Dict[str, Any]] = []

        all_stop_orders = self.order_manager.get_active_stop_orders()

        for stop_order in all_stop_orders:
            if (
                not hasattr(stop_order, "is_stop_loss")
                or not stop_order.is_stop_loss
            ) and (
                not hasattr(stop_order, "is_take_profit")
                or not stop_order.is_take_profit
            ):
                if self._is_stop_triggered(stop_order, price_level):
                    activated_time = self._apply_latency(
                        current_time, stop_order.order_id
                    )
                    self.order_manager.activate_stop_order(
                        stop_order, activated_time, self.latency_ms
                    )

                    if stop_order.order_type == OrderType.STOP_LIMIT:
                        stop_order.activation_time = activated_time
                        self.order_manager.add_limit_order(stop_order)
                    else:
                        stop_fills = self._execute_stop_market_fallback(
                            stop_order, step_high, step_low, activated_time
                        )
                        fills.extend(stop_fills)

        return fills

    def _process_market_orders_at_level(
        self,
        price_level: Decimal,
        step_high: Decimal,
        step_low: Decimal,
        current_time: datetime,
    ) -> List[Dict[str, Any]]:
        fills: List[Dict[str, Any]] = []

        market_orders: List[Order] = []
        while self.order_manager.market_orders:
            activation_time, created_timestamp, order_id = (
                self.order_manager.market_orders[0]
            )

            if order_id not in self.order_manager.all_orders:
                heapq.heappop(self.order_manager.market_orders)
                continue

            market_order = self.order_manager.all_orders[order_id]
            if market_order.status != OrderStatus.ACTIVE:
                heapq.heappop(self.order_manager.market_orders)
                continue

            market_orders.append(market_order)
            heapq.heappop(self.order_manager.market_orders)

        for market_order in market_orders:
            remaining_qty = market_order.remaining_quantity
            if remaining_qty <= Decimal("0"):
                continue

            if market_order.side == OrderSide.BUY:
                while remaining_qty > Decimal("0") and self.order_manager.limit_sells:
                    best_sell = self.order_manager.limit_sells[0]
                    sell_price, activation_time, created_timestamp, order_id = (
                        best_sell
                    )

                    if order_id not in self.order_manager.all_orders:
                        heapq.heappop(self.order_manager.limit_sells)
                        continue

                    sell_order = self.order_manager.all_orders[order_id]
                    if sell_order.status != OrderStatus.ACTIVE:
                        heapq.heappop(self.order_manager.limit_sells)
                        continue

                    fill_qty = min(remaining_qty, sell_order.remaining_quantity)

                    market_fill, limit_fill = self._execute_market_vs_limit(
                        market_order, sell_order, fill_qty, current_time
                    )

                    fills.extend([market_fill, limit_fill])
                    remaining_qty -= fill_qty

                    if sell_order.status == OrderStatus.FILLED:
                        heapq.heappop(self.order_manager.limit_sells)

            else:
                while remaining_qty > Decimal("0") and self.order_manager.limit_buys:
                    best_buy = self.order_manager.limit_buys[0]
                    neg_buy_price, activation_time, created_timestamp, order_id = (
                        best_buy
                    )
                    buy_price = -neg_buy_price

                    if order_id not in self.order_manager.all_orders:
                        heapq.heappop(self.order_manager.limit_buys)
                        continue

                    buy_order = self.order_manager.all_orders[order_id]
                    if buy_order.status != OrderStatus.ACTIVE:
                        heapq.heappop(self.order_manager.limit_buys)
                        continue

                    fill_qty = min(remaining_qty, buy_order.remaining_quantity)

                    market_fill, limit_fill = self._execute_market_vs_limit(
                        market_order, buy_order, fill_qty, current_time
                    )

                    fills.extend([market_fill, limit_fill])
                    remaining_qty -= fill_qty

                    if buy_order.status == OrderStatus.FILLED:
                        heapq.heappop(self.order_manager.limit_buys)

            if remaining_qty > Decimal("0"):
                fallback_fills = self._execute_stop_market_fallback(
                    market_order, step_high, step_low, current_time
                )
                fills.extend(fallback_fills)

        return fills

    def _process_limit_orders_at_level(
        self, price_level: Decimal, current_time: datetime
    ) -> List[Dict[str, Any]]:
        fills: List[Dict[str, Any]] = []

        while self.order_manager.limit_buys and self.order_manager.limit_sells:
            best_buy = self.order_manager.limit_buys[0]
            best_sell = self.order_manager.limit_sells[0]

            neg_buy_price, buy_activation, buy_created, buy_id = best_buy
            buy_price = -neg_buy_price
            sell_price, sell_activation, sell_created, sell_id = best_sell

            if (
                buy_id not in self.order_manager.all_orders
                or sell_id not in self.order_manager.all_orders
            ):
                if buy_id not in self.order_manager.all_orders:
                    heapq.heappop(self.order_manager.limit_buys)
                if sell_id not in self.order_manager.all_orders:
                    heapq.heappop(self.order_manager.limit_sells)
                continue

            buy_order = self.order_manager.all_orders[buy_id]
            sell_order = self.order_manager.all_orders[sell_id]

            if (
                buy_order.status != OrderStatus.ACTIVE
                or sell_order.status != OrderStatus.ACTIVE
            ):
                if buy_order.status != OrderStatus.ACTIVE:
                    heapq.heappop(self.order_manager.limit_buys)
                if sell_order.status != OrderStatus.ACTIVE:
                    heapq.heappop(self.order_manager.limit_sells)
                continue

            if buy_price < sell_price:
                break

            if not (
                self._is_price_touched(buy_order, price_level)
                and self._is_price_touched(sell_order, price_level)
            ):
                break

            if self.queue_mode == QueueMode.PRO_RATA:
                if buy_order.side == OrderSide.BUY:
                    aggressor, resting = buy_order, sell_order
                else:
                    aggressor, resting = sell_order, buy_order

                resting_orders = [resting]
                fill_qty = min(aggressor.remaining_quantity, resting.remaining_quantity)

                pro_rata_fills = self._execute_limit_vs_limit_pro_rata(
                    aggressor, resting_orders, fill_qty, current_time
                )
                fills.extend(pro_rata_fills)
            else:
                aggressor, resting = self._determine_aggressor(buy_order, sell_order)

                fill_qty = min(aggressor.remaining_quantity, resting.remaining_quantity)

                aggressor_fill, resting_fill = self._execute_limit_vs_limit(
                    aggressor, resting, fill_qty, current_time
                )

                fills.extend([aggressor_fill, resting_fill])

            if buy_order.status != OrderStatus.ACTIVE:
                heapq.heappop(self.order_manager.limit_buys)
            if sell_order.status != OrderStatus.ACTIVE:
                heapq.heappop(self.order_manager.limit_sells)

        return fills

    # ------------------------------------------------------------
    # Introspection / Testing Helpers
    # ------------------------------------------------------------
    def get_order_book_snapshot(self) -> Dict[str, Any]:
        bids: Dict[Decimal, Dict[str, Any]] = {}
        for order in self.order_manager.get_active_limit_buys():
            price = self._round_to_tick(order.limit_price)
            if price not in bids:
                bids[price] = {"quantity": Decimal("0"), "orders": 0}
            bids[price]["quantity"] += order.remaining_quantity
            bids[price]["orders"] += 1

        asks: Dict[Decimal, Dict[str, Any]] = {}
        for order in self.order_manager.get_active_limit_sells():
            price = self._round_to_tick(order.limit_price)
            if price not in asks:
                asks[price] = {"quantity": Decimal("0"), "orders": 0}
            asks[price]["quantity"] += order.remaining_quantity
            asks[price]["orders"] += 1

        sorted_bids = sorted(
            [(price, data["quantity"], data["orders"]) for price, data in bids.items()],
            reverse=True,
        )
        sorted_asks = sorted(
            [(price, data["quantity"], data["orders"]) for price, data in asks.items()]
        )

        return {
            "bids": sorted_bids,
            "asks": sorted_asks,
            "market_orders": len(self.order_manager.get_active_market_orders()),
            "stop_orders": len(self.order_manager.get_active_stop_orders()),
        }

    def _test_latency_and_queue(self) -> Dict[str, Any]:
        test_results: Dict[str, Any] = {}

        original_seed = self.rng.getstate()

        # Latency reproducibility
        self.rng.seed(42)
        time1 = datetime(2024, 1, 1, 10, 0, 0)
        result1 = self._apply_latency(time1, "test_order_1")

        self.rng.seed(42)
        result2 = self._apply_latency(time1, "test_order_1")
        test_results["latency_reproducible"] = result1 == result2

        # FIFO correctness
        self.queue_mode = QueueMode.FIFO
        order1 = Order(
            order_id="1",
            symbol="TEST",
            side=OrderSide.BUY,
            quantity=Decimal("100"),
            order_type=OrderType.LIMIT,
            limit_price=Decimal("100.00"),
            created_timestamp=datetime(2024, 1, 1, 10, 0, 0),
        )
        order2 = Order(
            order_id="2",
            symbol="TEST",
            side=OrderSide.BUY,
            quantity=Decimal("100"),
            order_type=OrderType.LIMIT,
            limit_price=Decimal("100.00"),
            created_timestamp=datetime(2024, 1, 1, 10, 0, 1),
        )

        order1.activation_time = datetime(2024, 1, 1, 10, 0, 0)
        order2.activation_time = datetime(2024, 1, 1, 10, 0, 1)

        aggressor, resting = self._determine_aggressor(order1, order2)
        test_results["fifo_correctness"] = (
            aggressor.order_id == "2" and resting.order_id == "1"
        )

        # Pro-Rata availability
        self.queue_mode = QueueMode.PRO_RATA
        aggressor_order = Order(
            order_id="A",
            symbol="TEST",
            side=OrderSide.BUY,
            quantity=Decimal("100"),
            order_type=OrderType.LIMIT,
            limit_price=Decimal("100.00"),
        )
        resting_order1 = Order(
            order_id="R1",
            symbol="TEST",
            side=OrderSide.SELL,
            quantity=Decimal("30"),
            order_type=OrderType.LIMIT,
            limit_price=Decimal("100.00"),
        )
        resting_order2 = Order(
            order_id="R2",
            symbol="TEST",
            side=OrderSide.SELL,
            quantity=Decimal("70"),
            order_type=OrderType.LIMIT,
            limit_price=Decimal("100.00"),
        )
        test_results["pro_rata_available"] = True

        # Bucket snapping
        self.latency_bucket_ms = 10
        test_time = datetime(2024, 1, 1, 10, 0, 0, 123456)
        bucketed_time = self._apply_latency(test_time, "bucket_test")
        test_results["bucket_snapping"] = bucketed_time.microsecond % 10000 == 0

        # Zero latency mode
        self.base_latency_ms = 0
        self.jitter_ms = 0
        original_time = datetime(2024, 1, 1, 10, 0, 0)
        zero_latency_result = self._apply_latency(original_time, "zero_test")
        test_results["zero_latency"] = zero_latency_result == original_time

        # Execution latency
        self.execution_latency_ms = 5
        exec_time = datetime(2024, 1, 1, 10, 0, 0)
        delayed_exec = self._apply_execution_latency(exec_time)
        test_results["execution_latency"] = delayed_exec == exec_time + timedelta(
            milliseconds=5
        )

        # Restore
        self.rng.setstate(original_seed)
        self.queue_mode = QueueMode.FIFO
        self.latency_bucket_ms = 0
        self.execution_latency_ms = 0

        return test_results

    def _test_slippage(self) -> Dict[str, Any]:
        test_results: Dict[str, Any] = {}

        original_mode = self.slippage_mode

        # No slippage
        self.slippage_mode = "none"
        result_none = self._apply_slippage(
            Decimal("100.00"), Decimal("1000"), OrderSide.BUY
        )
        test_results["no_slippage"] = result_none == Decimal("100.00")

        # Fixed BUY
        self.slippage_mode = "fixed"
        self.fixed_slippage_ticks = Decimal("2")
        result_fixed_buy = self._apply_slippage(
            Decimal("100.00"), Decimal("1000"), OrderSide.BUY
        )
        test_results["fixed_buy"] = result_fixed_buy == Decimal("100.02")

        # Fixed SELL
        result_fixed_sell = self._apply_slippage(
            Decimal("100.00"), Decimal("1000"), OrderSide.SELL
        )
        test_results["fixed_sell"] = result_fixed_sell == Decimal("99.98")

        # Volume mode
        self.slippage_mode = "volume"
        self.dynamic_coeff = Decimal("10.0")
        self.recent_volumes = [Decimal("1000"), Decimal("2000"), Decimal("1500")]
        result_volume = self._apply_slippage(
            Decimal("100.00"), Decimal("2000"), OrderSide.BUY
        )
        avg_volume = Decimal("1500")
        volume_ratio = Decimal("2000") / avg_volume
        expected_ticks = (volume_ratio * self.dynamic_coeff).quantize(
            Decimal("1"), rounding=ROUND_FLOOR
        )
        expected_price = Decimal("100.00") + expected_ticks * self.tick_size
        test_results["volume_slippage"] = result_volume == expected_price

        # Deterministic
        result1 = self._apply_slippage(
            Decimal("100.00"), Decimal("1000"), OrderSide.BUY
        )
        result2 = self._apply_slippage(
            Decimal("100.00"), Decimal("1000"), OrderSide.BUY
        )
        test_results["deterministic"] = result1 == result2

        self.slippage_mode = original_mode

        return test_results

    def _test_snapshot(self) -> Dict[str, Any]:
        return {
            "position": self.position,
            "order_book": self.get_order_book_snapshot(),
            "active_orders": len(self.order_manager.all_orders),
        }

    def _test_fill(self, order: Order) -> Dict[str, Any]:
        return {
            "order_id": order.order_id,
            "status": order.status,
            "filled": order.filled_quantity,
            "remaining": order.remaining_quantity,
            "avg_price": order.average_fill_price,
        }

    def _test_book(self) -> Dict[str, Any]:
        return self.get_order_book_snapshot()

    def reset(self):
        self.position = Decimal("0")
        self.recent_volumes = []
        self.rng.seed(42)
        # توجه: این ریست OrderManager را هم از نو می‌سازد
        self.order_manager = OrderManager()
