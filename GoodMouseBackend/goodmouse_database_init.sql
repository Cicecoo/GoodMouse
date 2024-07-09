CREATE DATABASE goodmouse_data;
use goodmouse_data;
CREATE TABLE mouse_usage(
    time_stamp DATETIME NOT NULL,
    using_seconds FLOAT NOT NULL
);
CREATE TABLE action_count(
    id INT PRIMARY KEY AUTO_INCREMENT,
    time_stamp DATETIME NOT NULL,
    action_type VARCHAR(20) NOT NULL,
    count INT NOT NULL
);