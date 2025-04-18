import random

def print_board(board):
    """Display the Tic-Tac-Toe board"""
    print(f" {board[0]} | {board[1]} | {board[2]} ")
    print("-----------")
    print(f" {board[3]} | {board[4]} | {board[5]} ")
    print("-----------")
    print(f" {board[6]} | {board[7]} | {board[8]} ")

def check_winner(board):
    """Check if there's a winner or if the game is a tie"""
    # All possible winning combinations
    win_patterns = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # columns
        [0, 4, 8], [2, 4, 6]              # diagonals
    ]
    
    for pattern in win_patterns:
        if board[pattern[0]] == board[pattern[1]] == board[pattern[2]] != " ":
            return board[pattern[0]]  # returns 'X' or 'O'
    
    if " " not in board:
        return "Tie"
    return None

def minimax(board, depth, is_maximizing, alpha=-float('inf'), beta=float('inf')):
    """Minimax algorithm with Alpha-Beta pruning"""
    result = check_winner(board)
    
    if result is not None:
        if result == "X":
            return -10 + depth
        elif result == "O":
            return 10 - depth
        else:
            return 0
    
    if is_maximizing:
        best_score = -float('inf')
        for i in range(9):
            if board[i] == " ":
                board[i] = "O"
                score = minimax(board, depth + 1, False, alpha, beta)
                board[i] = " "
                best_score = max(score, best_score)
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break
        return best_score
    else:
        best_score = float('inf')
        for i in range(9):
            if board[i] == " ":
                board[i] = "X"
                score = minimax(board, depth + 1, True, alpha, beta)
                board[i] = " "
                best_score = min(score, best_score)
                beta = min(beta, best_score)
                if beta <= alpha:
                    break
        return best_score

def ai_move(board):
    """Determine the best move for the AI using Minimax"""
    best_score = -float('inf')
    best_move = None
    
    for i in range(9):
        if board[i] == " ":
            board[i] = "O"
            score = minimax(board, 0, False)
            board[i] = " "
            if score > best_score:
                best_score = score
                best_move = i
                
    return best_move

def play_game():
    """Main game loop"""
    board = [" "] * 9
    current_player = "X"  # Human starts first
    
    print("Welcome to Tic-Tac-Toe!")
    print("You are X and the AI is O")
    print("Enter a number (1-9) to make your move:")
    print_board(["1", "2", "3", "4", "5", "6", "7", "8", "9"])
    print("\nLet's begin!\n")
    
    while True:
        if current_player == "X":
            # Human player's turn
            while True:
                try:
                    move = int(input("Your move (1-9): ")) - 1
                    if 0 <= move <= 8 and board[move] == " ":
                        break
                    else:
                        print("Invalid move. Try again.")
                except ValueError:
                    print("Please enter a number between 1 and 9.")
            
            board[move] = "X"
        else:
            # AI's turn
            print("\nAI is thinking...")
            move = ai_move(board)
            board[move] = "O"
            print(f"AI plays at position {move + 1}")
        
        print_board(board)
        
        winner = check_winner(board)
        if winner:
            if winner == "Tie":
                print("\nIt's a tie!")
            else:
                print(f"\n{'You' if winner == 'X' else 'AI'} wins!")
            break
        
        current_player = "O" if current_player == "X" else "X"

if __name__ == "__main__":
    play_game()