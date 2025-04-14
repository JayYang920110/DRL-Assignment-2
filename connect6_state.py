import sys
import numpy as np
import random
import copy


class Connect6State:
    """Connect6游戏状态，用于MCTS算法"""
    def __init__(self, board, turn, size=19):
        self.board = copy.deepcopy(board)
        self.turn = turn  # 1: Black, 2: White
        self.size = size
        self.availables = [(r, c) for r in range(size) for c in range(size) if board[r, c] == 0]
        self.move_history = []  # 可選：記錄所有已下棋子
        self.pending_moves = []  # 暫存這一手內的棋子（用來決定何時換人）

    def get_current_player(self):
        """返回当前玩家"""
        return self.turn
     
    def do_move(self, move):
        """在 Connect6 中下一顆棋子，pos 是 (r, c)，只接受單點行動"""
        r, c = move
        assert (r, c) in self.availables, "Invalid move"
        
        self.board[r, c] = self.turn
        self.availables.remove((r, c))
        self.pending_moves.append((r, c))  # 累積本手內下的棋子

        # 黑子第一手只下一顆（直接切換）
        # if len(self.move_history) == 0 and self.turn == 1:
        #     self.turn = 2
        #     self.pending_moves.clear()
        # 之後每手都要兩顆，才切換顏色
        if len(self.pending_moves) == 2:
            self.turn = 3 - self.turn
            self.pending_moves.clear()

        self.move_history.append((r, c))
    
    def game_end(self):
        """检查游戏是否结束
        返回: (is_end, winner), winner: 0-平局, 1-黑胜, 2-白胜
        """
        # 使用Connect6Game的check_win逻辑
        winner = self._check_win()
        if winner != 0:
            return True, winner
        
        # 检查棋盘是否已满
        if len(self.availables) < 2:
            return True, -1  # 平局
        
        return False, -1
    
    def _check_win(self):
        """检查是否有赢家"""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r, c] != 0:
                    current_color = self.board[r, c]
                    for dr, dc in directions:
                        prev_r, prev_c = r - dr, c - dc
                        if 0 <= prev_r < self.size and 0 <= prev_c < self.size and self.board[prev_r, prev_c] == current_color:
                            continue
                        count = 0
                        rr, cc = r, c
                        while 0 <= rr < self.size and 0 <= cc < self.size and self.board[rr, cc] == current_color:
                            count += 1
                            rr += dr
                            cc += dc
                        if count >= 6:
                            return current_color
        return 0