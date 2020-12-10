use pyo3::prelude::*;

/// Positions represented by flattened index. (y,x) = y*8+x
const SKIP_ACTION: usize = 65;


/// Get all possible places the given color can go
pub fn get_possible_actions(_stone: usize, _board: [[usize; 8]; 8]) -> Vec<usize> {
    let mut positions: Vec<usize> = Vec::with_capacity(64);

    for y in 0..8 {
        for x in 0..8 {
            if _board[y][x] == 0 && _board != flip(x, y, _stone, _board) {
                positions.push(y*8+x);
            }
        }
    }
    if positions.len() == 0 {
        positions.push(SKIP_ACTION)
    }

    positions
}

pub fn place_tile(_x: usize, _y: usize, _color: usize, _board: [[usize; 8]; 8]) -> [[usize; 8]; 8] { 
    let mut new_board = flip(_x, _y, _color, _board);
    new_board[_y][_x] = _color;
    // println!("new board:\n {:?}\nboard:\n {:?}\nplace(y,x): {}, {}",new_board, _board,_y,_x);
    new_board
}

/// Flip tiles if placing tile of given color at [x,y] on board. Returning new board
fn flip(_x: usize, _y: usize, _color: usize, _board: [[usize; 8]; 8]) -> [[usize; 8]; 8] {
    let opponent_color = if _color == 1 { 2 } else { 1 };
    let dx: [i32; 8] = [0, -1, -1, -1, 0, 1, 1, 1];
    let dy: [i32; 8] = [1, 1, 0, -1, -1, -1, 0, 1];
    let mut new_board = _board;

    for id in 0..8 {
        let mut _x_pos = _x as i32 + dx[id];
        let mut _y_pos = _y as i32 + dy[id];
        if _x_pos < 0 || _x_pos > 7 || _y_pos < 0 || _y_pos > 7 {
            continue;
        }

        if _board[_y_pos as usize][_x_pos as usize] == opponent_color {
            let mut flag = true;
            let mut count_max = 0;

            loop {
                count_max += 1;
                _x_pos += dx[id];
                _y_pos += dy[id];

                if _x_pos < 0
                    || _x_pos > 7
                    || _y_pos < 0
                    || _y_pos > 7
                    || _board[_y_pos as usize][_x_pos as usize] == 0
                {
                    flag = false;
                    break;
                } else if _board[_y_pos as usize][_x_pos as usize] == _color {
                    break;
                }
            }

            if flag {
                _x_pos = _x as i32;
                _y_pos = _y as i32;

                for _i in 0..count_max {
                    _x_pos += dx[id];
                    _y_pos += dy[id];
                    new_board[_y_pos as usize][_x_pos as usize] = _color;
                }
            }
        }
    }
    new_board
}

/// Get scores as [white, black]
pub fn scores(_board: [[usize; 8]; 8]) -> [usize; 2] {
    let mut stones: [usize; 2] = [0; 2];

    for i in 0..8 {
        for j in 0..8 {
            if _board[i][j] > 0 {
                stones[_board[i][j] - 1] += 1;
            }
        }
    }

    stones
}