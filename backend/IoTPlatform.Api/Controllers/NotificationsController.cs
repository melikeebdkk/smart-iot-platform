using Microsoft.AspNetCore.Mvc;
using Npgsql;
using System.Text.Json;

namespace IoTPlatform.Api.Controllers
{
    [ApiController]
    [Route("api/notifications")]
    public class NotificationsController : ControllerBase
    {
        private readonly string _connStr = "Host=localhost;Port=5432;Database=iotdb;Username=iotuser;Password=iotpass";

        [HttpGet("latest")]
        public IActionResult GetLatest([FromQuery] int minutes = 60)
        {
            using var conn = new NpgsqlConnection(_connStr);
            conn.Open();

            var cmd = new NpgsqlCommand(@"
                SELECT id, time, severity, source, device_id, type, message, payload, status
                FROM ai_notifications
                WHERE time >= NOW() - (@m || ' minutes')::interval
                ORDER BY time DESC
                LIMIT 200;", conn);

            cmd.Parameters.AddWithValue("m", minutes);

            using var reader = cmd.ExecuteReader();
            var list = new List<object>();

            while (reader.Read())
            {
                list.Add(new
                {
                    id = reader.GetInt64(0),
                    time = reader.GetDateTime(1),
                    severity = reader.GetString(2),
                    source = reader.GetString(3),
                    device_id = reader.IsDBNull(4) ? null : reader.GetString(4),
                    type = reader.GetString(5),
                    message = reader.IsDBNull(6) ? null : reader.GetString(6),
                    payload = reader.IsDBNull(7) ? null : (object)reader.GetFieldValue<JsonElement>(7),
                    status = reader.GetString(8)

                });
            }

            return Ok(list);
        }

        [HttpPost("{id}/ack")]
        public IActionResult Ack(long id)
        {
            using var conn = new NpgsqlConnection(_connStr);
            conn.Open();

            var cmd = new NpgsqlCommand("UPDATE ai_notifications SET status='ACK' WHERE id=@id;", conn);
            cmd.Parameters.AddWithValue("id", id);

            var n = cmd.ExecuteNonQuery();
            if (n == 0) return NotFound();

            return Ok(new { ok = true });
        }

        [HttpPost("{id}/resolve")]
        public IActionResult Resolve(long id)
        {
            using var conn = new NpgsqlConnection(_connStr);
            conn.Open();

            var cmd = new NpgsqlCommand("UPDATE ai_notifications SET status='RESOLVED' WHERE id=@id;", conn);
            cmd.Parameters.AddWithValue("id", id);

            var n = cmd.ExecuteNonQuery();
            if (n == 0) return NotFound();

            return Ok(new { ok = true });
        }
    }
}
