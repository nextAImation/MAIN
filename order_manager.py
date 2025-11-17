# order_manager.py
import heapq
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from decimal import Decimal
import uuid
from datetime import datetime, timedelta
from order_enums import OrderType, OrderSide, OrderStatus
class OrderType(Enum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"
    STOP_MARKET = "STOP_MARKET"
    STOP_LIMIT = "STOP_LIMIT"

class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderStatus(Enum):
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

@dataclass
class Fill:
    fill_price: Decimal
    fill_quantity: Decimal
    timestamp: datetime
    fee: Decimal
    is_maker: bool

@dataclass
class Order:
    order_id: str
    order_type: OrderType
    side: OrderSide
    symbol: str
    quantity: Decimal
    created_timestamp: datetime
    last_updated_timestamp: datetime
    filled_quantity: Decimal = Decimal('0')
    limit_price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    reduce_only: bool = False
    post_only: bool = False
    parent_order_id: Optional[str] = None
    activation_time: Optional[datetime] = None
    average_fill_price: Decimal = Decimal('0')
    fills: List[Fill] = field(default_factory=list)
    status: OrderStatus = OrderStatus.PENDING
    
    def __post_init__(self):
        if self.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and self.limit_price is None:
            raise ValueError("LIMIT and STOP_LIMIT orders must have limit_price")
        if self.order_type in [OrderType.STOP_MARKET, OrderType.STOP_LIMIT] and self.stop_price is None:
            raise ValueError("STOP orders must have stop_price")
        if self.quantity <= Decimal('0'):
            raise ValueError("Quantity must be positive")
    
    @property
    def remaining_quantity(self) -> Decimal:
        return self.quantity - self.filled_quantity
    
    def update_status(self, new_status: OrderStatus, timestamp: datetime) -> None:
        self.status = new_status
        self.last_updated_timestamp = timestamp
    
    def update_fill(self, fill_price: Decimal, fill_quantity: Decimal, timestamp: datetime, 
                   fee: Decimal, is_maker: bool) -> None:
        if fill_quantity > self.remaining_quantity:
            raise ValueError("Fill quantity exceeds remaining quantity")
        
        self.filled_quantity += fill_quantity
        total_value = self.average_fill_price * (self.filled_quantity - fill_quantity) + fill_price * fill_quantity
        self.average_fill_price = total_value / self.filled_quantity if self.filled_quantity > 0 else Decimal('0')
        
        self.fills.append(Fill(
            fill_price=fill_price,
            fill_quantity=fill_quantity,
            timestamp=timestamp,
            fee=fee,
            is_maker=is_maker
        ))
        
        self.last_updated_timestamp = timestamp
        
        if self.filled_quantity == self.quantity:
            self.status = OrderStatus.FILLED
        elif self.filled_quantity > Decimal('0'):
            self.status = OrderStatus.PARTIAL

class OrderManager:
    def __init__(self):
        self.limit_buys = []    # (-limit_price, activation_time, created_timestamp, order_id)
        self.limit_sells = []   # (limit_price, activation_time, created_timestamp, order_id)
        self.stop_buys = []     # (stop_price, activation_time, created_timestamp, order_id)
        self.stop_sells = []    # (-stop_price, activation_time, created_timestamp, order_id)
        self.market_orders = [] # (activation_time, created_timestamp, order_id)
        
        self.all_orders: Dict[str, Order] = {}
        self.oco_groups: Dict[str, List[str]] = {}
    
    def create_order(self, order_type: OrderType, side: OrderSide, symbol: str, quantity: Decimal,
                    limit_price: Optional[Decimal] = None, stop_price: Optional[Decimal] = None,
                    reduce_only: bool = False, post_only: bool = False, 
                    parent_order_id: Optional[str] = None, latency_ms: int = 0, 
                    current_time: datetime = None) -> Order:
        
        if current_time is None:
            current_time = datetime.now()
        
        order_id = str(uuid.uuid4())
        activation_time = current_time + timedelta(milliseconds=latency_ms)
        
        order = Order(
            order_id=order_id,
            order_type=order_type,
            side=side,
            symbol=symbol,
            quantity=quantity,
            created_timestamp=current_time,
            last_updated_timestamp=current_time,
            limit_price=limit_price,
            stop_price=stop_price,
            reduce_only=reduce_only,
            post_only=post_only,
            parent_order_id=parent_order_id,
            activation_time=activation_time,
            status=OrderStatus.PENDING
        )
        
        if parent_order_id:
            if parent_order_id not in self.oco_groups:
                self.oco_groups[parent_order_id] = []
            self.oco_groups[parent_order_id].append(order_id)
        
        self.all_orders[order_id] = order
        
        if latency_ms == 0:
            self._activate_pending_order(order_id, current_time)
        
        return order
    
    def _activate_pending_order(self, order_id: str, current_time: datetime) -> bool:
        if order_id not in self.all_orders:
            return False
        
        order = self.all_orders[order_id]
        
        if order.status != OrderStatus.PENDING:
            return False
        
        if order.activation_time > current_time:
            return False
        
        order.update_status(OrderStatus.ACTIVE, current_time)
        
        if order.order_type == OrderType.LIMIT:
            if order.side == OrderSide.BUY:
                heapq.heappush(self.limit_buys, (-order.limit_price, order.activation_time, order.created_timestamp, order_id))
            else:
                heapq.heappush(self.limit_sells, (order.limit_price, order.activation_time, order.created_timestamp, order_id))
        elif order.order_type == OrderType.MARKET:
            heapq.heappush(self.market_orders, (order.activation_time, order.created_timestamp, order_id))
        elif order.order_type == OrderType.STOP_MARKET:
            if order.side == OrderSide.BUY:
                heapq.heappush(self.stop_buys, (order.stop_price, order.activation_time, order.created_timestamp, order_id))
            else:
                heapq.heappush(self.stop_sells, (-order.stop_price, order.activation_time, order.created_timestamp, order_id))
        elif order.order_type == OrderType.STOP_LIMIT:
            if order.side == OrderSide.BUY:
                heapq.heappush(self.stop_buys, (order.stop_price, order.activation_time, order.created_timestamp, order_id))
            else:
                heapq.heappush(self.stop_sells, (-order.stop_price, order.activation_time, order.created_timestamp, order_id))
        
        return True
    
    def activate_stop_order(self, order: Order, current_time: datetime, latency_ms: int) -> bool:
        if order.order_type not in [OrderType.STOP_MARKET, OrderType.STOP_LIMIT]:
            return False
        
        if order.status != OrderStatus.ACTIVE:
            return False
        
        self._remove_from_heaps(order.order_id)
        
        activation_time = current_time + timedelta(milliseconds=latency_ms)
        order.activation_time = activation_time
        
        if order.order_type == OrderType.STOP_MARKET:
            order.order_type = OrderType.MARKET
        else:
            order.order_type = OrderType.LIMIT
            if order.limit_price is None:
                order.limit_price = order.stop_price
        
        if activation_time <= current_time:
            order.update_status(OrderStatus.ACTIVE, current_time)
            if order.order_type == OrderType.MARKET:
                heapq.heappush(self.market_orders, (order.activation_time, order.created_timestamp, order.order_id))
            else:
                if order.side == OrderSide.BUY:
                    heapq.heappush(self.limit_buys, (-order.limit_price, order.activation_time, order.created_timestamp, order.order_id))
                else:
                    heapq.heappush(self.limit_sells, (order.limit_price, order.activation_time, order.created_timestamp, order.order_id))
        else:
            order.update_status(OrderStatus.PENDING, current_time)
        
        return True
    
    def cancel_order(self, order_id: str, current_time: datetime) -> bool:
        if order_id not in self.all_orders:
            return False
        
        order = self.all_orders[order_id]
        
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            return False
        
        order.update_status(OrderStatus.CANCELLED, current_time)
        self._remove_from_heaps(order_id)
        self._handle_oco_on_terminal_state(order, current_time)
        
        return True
    
    def _handle_oco_on_terminal_state(self, order: Order, current_time: datetime) -> None:
        if not order.parent_order_id or order.parent_order_id not in self.oco_groups:
            return
        
        parent_id = order.parent_order_id
        for sibling_id in self.oco_groups[parent_id]:
            if sibling_id != order.order_id and sibling_id in self.all_orders:
                sibling = self.all_orders[sibling_id]
                if sibling.status not in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
                    sibling.update_status(OrderStatus.CANCELLED, current_time)
                    self._remove_from_heaps(sibling_id)
        
        del self.oco_groups[parent_id]
    
    def update_order_fill(self, order_id: str, fill_price: Decimal, fill_quantity: Decimal,
                         timestamp: datetime, fee_rate: Decimal, current_position: Decimal) -> bool:
        if order_id not in self.all_orders:
            return False
        
        order = self.all_orders[order_id]
        
        if order.status not in [OrderStatus.ACTIVE, OrderStatus.PARTIAL]:
            return False
        
        if not self._validate_reduce_only(order, current_position, fill_quantity):
            return False
        
        is_maker = order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]
        fee = fill_price * fill_quantity * fee_rate
        
        try:
            order.update_fill(fill_price, fill_quantity, timestamp, fee, is_maker)
            
            if order.status in [OrderStatus.FILLED, OrderStatus.REJECTED]:
                self._remove_from_heaps(order_id)
                self._handle_oco_on_terminal_state(order, timestamp)
            
            return True
        except ValueError:
            return False
    
    def _validate_reduce_only(self, order: Order, current_position: Decimal, fill_quantity: Decimal) -> bool:
        if not order.reduce_only:
            return True
        
        if order.side == OrderSide.BUY:
            new_position = current_position + fill_quantity
            return new_position <= Decimal('0')
        else:
            new_position = current_position - fill_quantity
            return new_position >= Decimal('0')
    
    def _remove_from_heaps(self, order_id: str) -> None:
        self.limit_buys = [item for item in self.limit_buys if item[3] != order_id]
        self.limit_sells = [item for item in self.limit_sells if item[3] != order_id]
        self.stop_buys = [item for item in self.stop_buys if item[3] != order_id]
        self.stop_sells = [item for item in self.stop_sells if item[3] != order_id]
        self.market_orders = [item for item in self.market_orders if item[2] != order_id]
        
        heapq.heapify(self.limit_buys)
        heapq.heapify(self.limit_sells)
        heapq.heapify(self.stop_buys)
        heapq.heapify(self.stop_sells)
        heapq.heapify(self.market_orders)
    
    def get_triggerable_stop_orders(self, high: Decimal, low: Decimal, current_time: datetime) -> List[Order]:
        candidates = []
        
        for heap in [self.stop_buys, self.stop_sells]:
            for stop_price, activation_time, created_timestamp, order_id in heap:
                if order_id not in self.all_orders:
                    continue
                
                order = self.all_orders[order_id]
                if order.status != OrderStatus.ACTIVE:
                    continue
                
                if order.activation_time > current_time:
                    continue
                
                if (order.side == OrderSide.BUY and high >= order.stop_price) or \
                   (order.side == OrderSide.SELL and low <= order.stop_price):
                    candidates.append(order)
        
        candidates.sort(key=lambda o: (o.activation_time, o.created_timestamp, o.order_id))
        return candidates
    
    def get_active_stop_orders(self) -> List[Order]:
        active_stops = []
        current_time = datetime.now()
        
        for heap in [self.stop_buys, self.stop_sells]:
            for stop_price, activation_time, created_timestamp, order_id in heap:
                if order_id not in self.all_orders:
                    continue
                
                order = self.all_orders[order_id]
                if (order.status == OrderStatus.ACTIVE and 
                    order.order_type in [OrderType.STOP_MARKET, OrderType.STOP_LIMIT] and
                    order.activation_time <= current_time):
                    active_stops.append(order)
        
        active_stops.sort(key=lambda o: (o.activation_time, o.created_timestamp, o.order_id))
        return active_stops
    
    def pop_best_limit_buy(self) -> Optional[Order]:
        while self.limit_buys:
            neg_price, activation_time, created_timestamp, order_id = self.limit_buys[0]
            if order_id not in self.all_orders:
                heapq.heappop(self.limit_buys)
                continue
            
            order = self.all_orders[order_id]
            if order.status == OrderStatus.ACTIVE:
                heapq.heappop(self.limit_buys)
                return order
            else:
                heapq.heappop(self.limit_buys)
        
        return None
    
    def pop_best_limit_sell(self) -> Optional[Order]:
        while self.limit_sells:
            price, activation_time, created_timestamp, order_id = self.limit_sells[0]
            if order_id not in self.all_orders:
                heapq.heappop(self.limit_sells)
                continue
            
            order = self.all_orders[order_id]
            if order.status == OrderStatus.ACTIVE:
                heapq.heappop(self.limit_sells)
                return order
            else:
                heapq.heappop(self.limit_sells)
        
        return None
    
    def pop_next_market(self) -> Optional[Order]:
        while self.market_orders:
            activation_time, created_timestamp, order_id = self.market_orders[0]
            if order_id not in self.all_orders:
                heapq.heappop(self.market_orders)
                continue
            
            order = self.all_orders[order_id]
            if order.status == OrderStatus.ACTIVE:
                heapq.heappop(self.market_orders)
                return order
            else:
                heapq.heappop(self.market_orders)
        
        return None
    
    def get_active_limit_buys(self) -> List[Order]:
        return [self.all_orders[order_id] for neg_price, activation_time, created_timestamp, order_id in self.limit_buys 
                if order_id in self.all_orders and self.all_orders[order_id].status == OrderStatus.ACTIVE]
    
    def get_active_limit_sells(self) -> List[Order]:
        return [self.all_orders[order_id] for price, activation_time, created_timestamp, order_id in self.limit_sells 
                if order_id in self.all_orders and self.all_orders[order_id].status == OrderStatus.ACTIVE]
    
    def get_active_market_orders(self) -> List[Order]:
        return [self.all_orders[order_id] for activation_time, created_timestamp, order_id in self.market_orders 
                if order_id in self.all_orders and self.all_orders[order_id].status == OrderStatus.ACTIVE]
    
    def process_pending_orders(self, current_time: datetime) -> None:
        pending_orders = [order_id for order_id, order in self.all_orders.items() 
                         if order.status == OrderStatus.PENDING and order.activation_time <= current_time]
        
        for order_id in pending_orders:
            self._activate_pending_order(order_id, current_time)
    
    def cleanup_completed_orders(self) -> None:
        completed_ids = [order_id for order_id, order in self.all_orders.items() 
                        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]]
        
        for order_id in completed_ids:
            order = self.all_orders[order_id]
            if order.parent_order_id is not None:
                self._handle_oco_on_terminal_state(order, order.last_updated_timestamp)
        
        for order_id in completed_ids:
            self._remove_from_heaps(order_id)
            if order_id in self.all_orders:
                del self.all_orders[order_id]
        
        expired_oco_groups = [parent_id for parent_id, siblings in self.oco_groups.items() 
                             if all(sibling_id not in self.all_orders for sibling_id in siblings)]
        
        for parent_id in expired_oco_groups:
            del self.oco_groups[parent_id]

# CHANGELOG (Stage 7)
# - Fixed get_active_stop_orders to be canonical source for MatchingEngine
# - Ensured get_active_stop_orders returns only ACTIVE stop orders with proper filtering
# - Added proper sorting by (activation_time, created_timestamp, order_id)
# - Verified activate_stop_order correctly removes from stop heaps and requeues
# - Confirmed cleanup_completed_orders properly removes inactive stop orders